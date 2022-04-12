import pdb
import torch.nn as nn
import torch
import models.GAN_models as GAN_models
from utils.load_checkpoint import load_checkpoint
from transforms.transform_data import transform_data
from utils.seed_everything import seed_everything
import networkx as nx
from utils.graph_utils import get_graph_data, get_adjacency_matrix, \
    create_graph_from_data
from inference.maximum_a_posteriori import compute_MAP
from inference.MCMC import hamiltonian_MC
import hamiltorch
from utils.compute_statistics import get_statistics_from_latent_samples, \
    get_demand_statistics
from plotting import plot_results
import numpy as np
import matplotlib.pyplot as plt
from utils.observation import observation_operator, get_test_observations
import wntr
from utils.graph_utils import get_incidence_mat, incidence_to_adjacency
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import ray

torch.set_default_dtype(torch.float32)


def get_obs_operators(obs_idx):
    flow_rate_obs_operator = lambda obs: observation_operator(data=obs,
                                                              obs_idx=obs_idx[
                                                                  'edge_obs_idx'],
                                                              variable='flow_rate')

    head_obs_operator = lambda obs: observation_operator(data=obs,
                                                         obs_idx=obs_idx[
                                                             'node_obs_idx'],
                                                         variable='head')

    obs_operator = lambda obs: observation_operator(data=obs,
                                                    obs_idx=obs_idx,
                                                    variable='both')

    return obs_operator, flow_rate_obs_operator, head_obs_operator


def get_MCGAN_results(obs_operator,
                      full_observations,
                      noise_distribution,
                      generator,
                      transformer,
                      gan_with_leak):

    z_init = torch.randn(1, latent_dim, requires_grad=True, device=device)
    z_map = compute_MAP(z=z_init,
                        observations=full_observations,
                        generator=generator,
                        obs_operator=obs_operator,
                        num_iter=500,
                        only_sensors=True)

    posterior_params = {'generator': generator,
                        'obs_operator': obs_operator,
                        'observations': full_observations,
                        'prior_mean': torch.zeros(latent_dim, device=device),
                        'prior_std': torch.ones(latent_dim, device=device),
                        'noise_mean': torch.cat(
                                [noise_distribution['flow_rate_noise_mean'],
                                 noise_distribution['head_noise_mean']],
                                dim=1),
                        'noise_std': torch.cat(
                                [noise_distribution['flow_rate_noise_std'],
                                 noise_distribution['head_noise_std']],
                                dim=1)}
    HMC_params = {'num_samples': 5000,
                  'step_size': 1.,
                  'num_steps_per_sample': 5,
                  'burn': 3500,
                  'integrator': hamiltorch.Integrator.IMPLICIT,
                  'sampler': hamiltorch.Sampler.HMC_NUTS,
                  'desired_accept_rate': 0.3}

    z_samples = hamiltonian_MC(z_init=torch.squeeze(z_map),
                               posterior_params=posterior_params,
                               HMC_params=HMC_params,
                               only_sensors=True)

    generated_samples = generator(z_samples)

    MCGAN_results = {}

    num_flow_sensors = full_observations.size(1)//2
    num_head_sensors = full_observations.size(1)//2

    flow_rate = {}
    flow_rate['mean'] = torch.mean(generated_samples[:,0:num_flow_sensors], dim=0)
    flow_rate['std'] = torch.std(generated_samples[:,0:num_flow_sensors], dim=0)

    head = {}
    head['mean'] = torch.mean(generated_samples[:,num_flow_sensors:num_flow_sensors+num_head_sensors], dim=0)
    head['std'] = torch.std(generated_samples[:,num_flow_sensors:num_flow_sensors+num_head_sensors], dim=0)

    MCGAN_results['flow_rate'] = flow_rate
    MCGAN_results['head'] = head

    if gan_with_leak:
        leak = {'demand_mean': torch.mean(generated_samples[:, num_flow_sensors+num_head_sensors],dim=0),
                'samples': generated_samples[:, -34:]}
        MCGAN_results['leak'] = leak

    return MCGAN_results


def get_errors(full_observations,
               MCGAN_results,
               edge_obs_idx,
               node_obs_idx):

    MCMC_obs_error = {}

    flow_rate_diff = torch.abs(full_observations[:, 0:len(edge_obs_idx)] \
                               - MCGAN_results['flow_rate']['mean'].unsqueeze(dim=0))
    MCMC_obs_error['flow_rate'] = torch.divide(flow_rate_diff,
                                               torch.abs(full_observations[:, 0:len(edge_obs_idx)]))


    head_diff = torch.abs(full_observations[:, -len(node_obs_idx):] \
                               - MCGAN_results['head']['mean'].unsqueeze(dim=0))
    MCMC_obs_error['head'] = torch.divide(head_diff,
                                           torch.abs(full_observations[:, -len(node_obs_idx):]))



    #MCMC_obs_error['flow_rate'] = torch.linalg.norm(full_observations[:, 0:len(edge_obs_idx)] \
    #                                                - MCGAN_results['flow_rate']['mean'].unsqueeze(dim=0))  \
    #                                                / torch.linalg.norm(full_observations[:, 0:len(edge_obs_idx)])
    #MCMC_obs_error['head'] = \
    #    torch.linalg.norm(full_observations[:, -len(node_obs_idx):] \
    #                      - MCGAN_results['head']['mean'].unsqueeze(dim=0)) \
    #                     / torch.linalg.norm(full_observations[:, -len(node_obs_idx):])

    return MCMC_obs_error

@ray.remote
def generate_train_data(case):

    true_leak.append(data_with_leak)

    if train_data:
        if data_with_leak:
            if small_leak:
                data_path = 'data/training_data_with_leak_small/network_' \
                            + str(
                        case)
            else:
                if small_demand_variance_data:
                    data_path = \
                        'data/training_data_with_leak_small_demand_variance/network_' + str(
                                case)
                else:
                    data_path = 'data/training_data_with_leak/network_' + \
                                str(
                                        case)
        else:
            if small_demand_variance_data:
                data_path = \
                    'data/training_data_no_leak_small_demand_variance' \
                    '/network_' + str(
                            case)
            else:
                data_path = 'data/training_data_no_leak/network_' + str(
                        case)
    else:
        if data_with_leak:
            if small_leak:
                data_path = 'data/test_data_with_leak_small/network_' + str(
                        case)
            else:
                if small_demand_variance_data:
                    data_path = \
                        'data/test_data_with_leak_small_demand_variance' \
                        '/network_' + str(
                                case)
                else:
                    data_path = 'data/test_data_with_leak/network_' + str(
                            case)
        else:
            if small_demand_variance_data:
                data_path = \
                    'data/test_data_no_leak_small_demand_variance' \
                    '/network_' + str(
                            case)
            else:
                data_path = 'data/test_data_no_leak/network_' + str(case)

    true_data_dict = nx.read_gpickle(data_path)
    G_true = true_data_dict['graph']
    true_flow_rate = torch.tensor(true_data_dict['flow_rate'].values,
                                  dtype=torch.get_default_dtype(),
                                  device=device)
    true_head = torch.tensor(true_data_dict['head'].values,
                             dtype=torch.get_default_dtype(),
                             device=device)
    true_demand = torch.tensor(true_data_dict['demand'].values,
                               dtype=torch.get_default_dtype(),
                               device=device)

    edge_obs_idx = [1, 5, 10, 15, 20, 25, 30]
    node_obs_idx = [1, 5, 10, 15, 20, 25, 30]
    obs_idx = {'edge_obs_idx': edge_obs_idx,
               'node_obs_idx': node_obs_idx}

    edge_obs_std = 0.05
    node_obs_std = 0.05
    std = {'edge_obs_std': edge_obs_std,
           'node_obs_std': node_obs_std}

    obs_operator, flow_rate_obs_operator, head_obs_operator = \
        get_obs_operators(
                obs_idx)

    full_observations, noise_distribution = get_test_observations(
            data_dict=true_data_dict,
            obs_idx=obs_idx,
            std=std,
            split=False,
            device=device)

    MCGAN_results = get_MCGAN_results(obs_operator,
                                      full_observations,
                                      noise_distribution,
                                      generator,
                                      transformer,
                                      gan_with_leak)
    MCMC_obs_error = get_errors(full_observations,
                               MCGAN_results,
                               edge_obs_idx,
                               node_obs_idx)

    flow_rate_std = torch.mean(MCGAN_results['flow_rate']['std']).item()
    head_std = torch.mean(MCGAN_results['head']['std']).item()

    if gan_with_leak:
        critic_input_leak = torch.cat([MCGAN_results['flow_rate']['mean'],
                                       MCGAN_results['head']['mean'],
                                       MCGAN_results['leak']['demand_mean'].unsqueeze(0),
                                       MCGAN_results['leak'][
                                           'samples'].mean(dim=0)])
        critic_input_no_leak = torch.cat(
                [MCGAN_results['flow_rate']['mean'],
                 MCGAN_results['head']['mean']])


    else:
        critic_input_leak = torch.cat([MCGAN_results['flow_rate']['mean'],
                                       MCGAN_results['head']['mean'],
                                       torch.tensor([0]),
                                       1 / 34 * torch.ones(34)])
        critic_input_leak_true = torch.cat([full_observations[0],
                                       torch.tensor([0]),
                                       1 / 34 * torch.ones(34)])

        critic_input_no_leak = torch.cat(
                [MCGAN_results['flow_rate']['mean'],
                 MCGAN_results['head']['mean']])

    def get_feat_vector(input, model):

        with torch.no_grad():
            my_output = None

            def my_hook(module_, input_, output_):
                nonlocal my_output
                my_output = output_

            a_hook = model.hidden_layers[-1].register_forward_hook(my_hook)
            model(input)
            a_hook.remove()
            return model.activation(my_output)

    #critic_score_leak = critic_leak(critic_input_leak)
    critic_out_gen = get_feat_vector(critic_input_leak, critic_leak)
    critic_out_true = get_feat_vector(critic_input_leak_true, critic_leak)
    critic_score_leak = torch.norm(critic_out_gen-critic_out_true)

    critic_out_gen = get_feat_vector(critic_input_no_leak, critic_no)
    critic_out_true = get_feat_vector(full_observations, critic_no)
    critic_score_no_leak = torch.norm(critic_out_gen-critic_out_true)

    '''
    return_list = {'flow_rate_obs_error': MCMC_obs_error['flow_rate'].item(),
                   'head_obs_error': MCMC_obs_error['head'].item(),
                   'flow_rate_std': flow_rate_std,
                   'head_std': head_std,
                   'critic_score_leak': critic_score_leak.item(),
                   'critic_score_no_leak': critic_score_no_leak.item(),
                   }
    '''

    return_list = {'flow_rate_std': flow_rate_std,
                   'head_std': head_std,
                   'critic_score_leak': critic_score_leak.item(),
                   'critic_score_no_leak': critic_score_no_leak.item(),
                   }
    flow_rate_error = MCMC_obs_error['flow_rate'].detach().numpy()
    head_error = MCMC_obs_error['head'].detach().numpy()
    for i in range(flow_rate_error.shape[1]):
        return_list[f'flow_rate_error_{i}'] = flow_rate_error[0, i]
    for i in range(head_error.shape[1]):
        return_list[f'head_error_{i}'] = head_error[0, i]

    flow_rate_std = MCGAN_results['flow_rate']['std'].detach().numpy()
    head_error_std = MCGAN_results['flow_rate']['std'].detach().numpy()
    for i in range(flow_rate_std.shape[0]):
        return_list[f'flow_rate_std_{i}'] = flow_rate_std[i]
    for i in range(head_error_std.shape[0]):
        return_list[f'head_std_{i}'] = head_error_std[i]

    print(f'{case} of {cases[-1]}')

    if data_with_leak:
        return_list['leak'] = 1
    else:
        return_list['leak'] = 0

    return return_list

if __name__ == "__main__":

    seed_everything()

    cuda = False
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Running on {device}')


    data_with_leak = False
    gan_with_leak = False

    cases = range(0, 1000)
    ray.init(num_cpus=15, log_to_driver=False)

    train_data = False

    small_leak = False
    small_demand_variance_data = False
    small_demand_variance_gan = False
    train_with_physics_loss = False

    if gan_with_leak:
        if small_leak:
            load_string = 'model_weights/GAN_small_leak'
        else:
            if small_demand_variance_gan:
                load_string = 'model_weights/GAN_leak_small_demand_variance'
            else:
                load_string = 'model_weights/GAN_leak'
            if train_with_physics_loss:
                load_string = 'model_weights/GAN_leak_and_pysics_loss'
    else:
        if small_demand_variance_gan:
            load_string = 'model_weights/GAN_no_leak_small_demand_variance'
        else:
            load_string = 'model_weights/GAN_no_leak'

    load_string += '_sensors'

    latent_dim = 8
    activation = nn.LeakyReLU()
    transformer = None  # transform_data(a=-1, b=1,
    #       leak=gan_with_leak,
    #       small=small_leak)

    sensors = {
        'flow_rate_sensors': [3, 7, 16, 19, 26, 29, 32],
        'head_sensors': [3, 8, 17, 19, 26, 27, 31]}

    output_dim = len(sensors['flow_rate_sensors']) + len(
            sensors['head_sensors'])

    generator_params = {'latent_dim': latent_dim,
                        'par_dim': 35,
                        'output_dim': output_dim,
                        'activation': activation,
                        'n_neurons': [8, 12, 16, 24],
                        'leak': gan_with_leak}

    generator = GAN_models.Generator(**generator_params).to(device)
    load_checkpoint(load_string, generator)
    generator.eval()

    n_neurons = [24, 16, 12, 8]
    critic_params_leak = {'activation': activation,
                          'n_neurons': n_neurons}
    critic_params_no_leak = {'activation': activation,
                             'n_neurons': n_neurons}
    critic_params_leak['input_dim'] = generator_params['output_dim']+generator_params['par_dim']
    critic_params_no_leak['input_dim'] = generator_params['output_dim']
    if train_with_physics_loss:
        critic_params_leak['input_dim'] += 32
        critic_params_no_leak['input_dim'] += 32

    generator = GAN_models.Generator(**generator_params).to(device)
    load_checkpoint(load_string, generator=generator)
    generator.eval()
    if small_leak:
        critic_load_string_leak = 'model_weights/GAN_small_leak'
    else:
        if small_demand_variance_gan:
            critic_load_string_leak = \
                'model_weights/GAN_leak_small_demand_variance'
        else:
            critic_load_string_leak = 'model_weights/GAN_leak'

    if small_demand_variance_gan:
        critic_load_string_no_leak = \
            'model_weights/GAN_no_leak_small_demand_variance'
    else:
        critic_load_string_no_leak = 'model_weights/GAN_no_leak'

    critic_load_string_leak += '_sensors'
    critic_load_string_no_leak += '_sensors'

    critic_leak = GAN_models.Critic(**critic_params_leak).to(device)
    load_checkpoint(critic_load_string_leak, critic=critic_leak)
    critic_leak.eval()

    critic_no = GAN_models.Critic(**critic_params_no_leak).to(device)
    load_checkpoint(critic_load_string_no_leak, critic=critic_no)
    critic_no.eval()

    num_cases = len(cases)
    true_leak = []
    pred_leak = []


    classification_data_ids = []
    for case in cases:
        classification_data_ids.append(generate_train_data.remote(case))

    classification_data = ray.get(classification_data_ids)  # [0, 1, 2, 3]
    #classification_data = generate_train_data(1)
    #classification_data = []
    #for case in cases:
    #    classification_data.append(generate_train_data(case))

    df = pd.DataFrame(columns=list(classification_data[0].keys()))
    for i in range(len(cases)):
        df = df.append(classification_data[i], ignore_index=True)

    print('DONE!!!!!!')

    if train_data:
        if data_with_leak:
            df.to_csv("training_classification_data_with_leak_sensors.csv")
        else:
            df.to_csv("training_classification_data_no_leak_sensors.csv")
    else:
        if data_with_leak:
            df.to_csv("test_classification_data_with_leak_sensors.csv")
        else:
            df.to_csv("test_classification_data_no_leak_sensors.csv")


