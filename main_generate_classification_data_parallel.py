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
                        num_iter=2500)

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
    HMC_params = {'num_samples': 2500,
                  'step_size': 1.,
                  'num_steps_per_sample': 5,
                  'burn': 1500,
                  'integrator': hamiltorch.Integrator.IMPLICIT,
                  'sampler': hamiltorch.Sampler.HMC_NUTS,
                  'desired_accept_rate': 0.3}

    z_samples = hamiltonian_MC(z_init=torch.squeeze(z_map),
                               posterior_params=posterior_params,
                               HMC_params=HMC_params)

    MCGAN_results = \
        get_statistics_from_latent_samples(z_samples=z_samples,
                                           generator=generator,
                                           gan_with_leak=gan_with_leak,
                                           transform=transformer.min_max_inverse_transform)

    return MCGAN_results


def get_errors(true_flow_rate,
               true_head,
               full_observations,
               flow_rate_obs_operator,
               head_obs_operator,
               MCGAN_results,
               edge_obs_idx,
               node_obs_idx):
    MCMC_obs_error = {}
    MCMC_full_error = {}

    MCMC_obs_error['flow_rate'] = \
        torch.linalg.norm(full_observations[:, 0:len(edge_obs_idx)] \
                          - flow_rate_obs_operator(
                MCGAN_results['flow_rate']['mean'].unsqueeze(dim=0))) \
        / torch.linalg.norm(full_observations[:, 0:len(edge_obs_idx)])
    MCMC_obs_error['head'] = \
        torch.linalg.norm(full_observations[:, -len(node_obs_idx):] \
                          - head_obs_operator(
                MCGAN_results['head']['mean'].unsqueeze(dim=0))) \
        / torch.linalg.norm(full_observations[:, -len(node_obs_idx):])

    MCMC_full_error['flow_rate'] = torch.linalg.norm(true_flow_rate - \
                                                     MCGAN_results['flow_rate'][
                                                         'mean'].unsqueeze(
                                                             dim=0)) \
                                   / torch.linalg.norm(true_flow_rate)
    MCMC_full_error['head'] = torch.linalg.norm(true_head - \
                                                MCGAN_results['head'][
                                                    'mean'].unsqueeze(dim=0)) \
                              / torch.linalg.norm(true_head)

    return MCMC_obs_error, MCMC_full_error

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

    MCMC_obs_error, MCMC_full_error = get_errors(true_flow_rate,
                                               true_head,
                                               full_observations,
                                               flow_rate_obs_operator,
                                               head_obs_operator,
                                               MCGAN_results,
                                               edge_obs_idx,
                                               node_obs_idx)

    flow_rate_std = torch.mean(MCGAN_results['flow_rate']['std']).item()
    head_std = torch.mean(MCGAN_results['head']['std']).item()

    MCGAN_results['demand'] = get_demand_statistics(data=MCGAN_results,
                                                    incidence_mat=incidence_mat)

    demand_diff = true_demand[0, 0] - \
                  MCGAN_results['demand']['mean'].detach()[0]
    demand_diff = demand_diff.item()

    transformer_leak = transform_data(a=-1, b=1,
                                      leak=True,
                                      small=small_leak)

    transformer_no_leak = transform_data(a=-1, b=1,
                                         leak=False,
                                         small=small_leak)
    if gan_with_leak:
        critic_input_leak = torch.cat([MCGAN_results['flow_rate']['mean'],
                                       MCGAN_results['head']['mean'],
                                       MCGAN_results['leak']['demand_mean'],
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
        critic_input_no_leak = torch.cat(
                [MCGAN_results['flow_rate']['mean'],
                 MCGAN_results['head']['mean']])

        critic_input_leak = transformer_leak.min_max_transform(
                critic_input_leak)
        critic_input_no_leak = transformer_no_leak.min_max_transform(
                critic_input_no_leak)

    if train_with_physics_loss:
        critic_input_leak = torch.cat([critic_input_leak,
                                       MCGAN_results['demand']['mean']])
        critic_input_no_leak = torch.cat([critic_input_no_leak,
                                          MCGAN_results['demand']['mean']])

    critic_input_leak = transformer_leak.min_max_transform(
            critic_input_leak)
    critic_input_no_leak = transformer_no_leak.min_max_transform(
            critic_input_no_leak)

    critic_score_leak = critic_leak(critic_input_leak)

    critic_score_no_leak = critic_no(critic_input_no_leak)

    return_list = {'demand_diff': demand_diff,
                   'flow_rate_obs_error': MCMC_obs_error[
                       'flow_rate'].item(),
                   'head_obs_error': MCMC_obs_error['head'].item(),
                   'flow_rate_std': flow_rate_std,
                   'head_std': head_std,
                   'critic_score_leak': critic_score_leak.item(),
                   'critic_score_no_leak': critic_score_no_leak.item(),
                   }

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

    inp_file = 'Input_files_EPANET/Hanoi_base_demand.inp'
    wn = wntr.network.WaterNetworkModel(inp_file)
    incidence_mat = get_incidence_mat(wn)
    adjacency_mat = incidence_to_adjacency(incidence_mat)

    incidence_mat = torch.tensor(incidence_mat, dtype=torch.get_default_dtype())
    adjacency_mat = torch.tensor(adjacency_mat, dtype=torch.get_default_dtype())

    data_with_leak = False
    gan_with_leak = False

    train_data = False

    small_leak = False
    small_demand_variance_data = False
    small_demand_variance_gan = False
    train_with_physics_loss = True

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

    latent_dim = 32
    activation = nn.Tanh()
    transformer = transform_data(a=-1, b=1,
                                 leak=gan_with_leak,
                                 small=small_leak)

    generator_params = {'latent_dim': latent_dim,
                        'par_dim': 35,
                        'output_dim': 66,
                        'activation': activation,
                        'n_neurons': [32, 40, 48, 56, 64],
                        'leak': gan_with_leak}

    n_neurons = [128, 112, 96, 80, 64, 48, 32, 16]
    critic_params_leak = {'activation': activation,
                          'n_neurons': n_neurons}
    critic_params_no_leak = {'activation': activation,
                             'n_neurons': n_neurons}
    critic_params_leak['input_dim'] = generator_params['output_dim'] + \
                                      generator_params['par_dim']
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

    if train_with_physics_loss:
        critic_load_string_leak = 'model_weights/GAN_leak_and_pysics_loss'

    critic_leak = GAN_models.Critic(**critic_params_leak).to(device)
    load_checkpoint(critic_load_string_leak, critic=critic_leak)
    critic_leak.eval()

    critic_no = GAN_models.Critic(**critic_params_no_leak).to(device)
    load_checkpoint(critic_load_string_no_leak, critic=critic_no)
    critic_no.eval()

    cases = range(0, 500)
    num_cases = len(cases)
    true_leak = []
    pred_leak = []

    leak_location_error = []
    obs_error_list = []
    crtic_score_list_leak = []
    crtic_score_list_no_leak = []
    reservoir_demand_diff = []
    flow_rate_std_list = []
    head_std_list = []
    flow_rate_obs_error_list = []
    head_obs_error_list = []
    leak_list = []

    ray.init(num_cpus=15, log_to_driver=False)

    classification_data_ids = []
    for case in cases:
        classification_data_ids.append(generate_train_data.remote(case))

    classification_data = ray.get(classification_data_ids)  # [0, 1, 2, 3]
    #classification_data = generate_train_data(1)

    for i in range(len(cases)):
        crtic_score_list_leak.append(classification_data[i]['critic_score_leak'])
        crtic_score_list_no_leak.append(classification_data[i]['critic_score_no_leak'])
        reservoir_demand_diff.append(classification_data[i]['demand_diff'])
        flow_rate_obs_error_list.append(classification_data[i]['flow_rate_obs_error'])
        head_obs_error_list.append(classification_data[i]['head_obs_error'])
        flow_rate_std_list.append(classification_data[i]['flow_rate_std'])
        head_std_list.append(classification_data[i]['head_std'])
        leak_list.append(classification_data[i]['leak'])

    flow_rate_obs_error_list = np.asarray(flow_rate_obs_error_list)
    head_obs_error_list = np.asarray(head_obs_error_list)
    crtic_score_list_leak = np.asarray(crtic_score_list_leak)
    crtic_score_list_no_leak = np.asarray(crtic_score_list_no_leak)
    reservoir_demand_diff = np.asarray(reservoir_demand_diff)
    flow_rate_std_list = np.asarray(flow_rate_std_list)
    head_std_list = np.asarray(head_std_list)
    leak_list = np.asarray(leak_list)

    numpy_data = np.stack((flow_rate_obs_error_list,
                           head_obs_error_list,
                           crtic_score_list_leak,
                           crtic_score_list_no_leak,
                           reservoir_demand_diff,
                           flow_rate_std_list,
                           head_std_list,
                           leak_list)
                          )
    numpy_data = np.transpose(numpy_data)

    column_names = ['flow_rate_obs_error',
                    'head_obs_error',
                    'critic_score_leak',
                    'critic_score_no_leak',
                    'reservoir_demand_diff',
                    'flow_rate_std',
                    'head_std',
                    'leak']

    df = pd.DataFrame(data=numpy_data, index=cases, columns=column_names)
    if train_data:
        if data_with_leak:
            df.to_pickle("training_classification_data_with_leak1.pkl")
        else:
            df.to_pickle("training_classification_data_no_leak1.pkl")
    else:
        if data_with_leak:
            df.to_pickle("test_classification_data_with_leak1.pkl")
        else:
            df.to_pickle("test_classification_data_no_leak1.pkl")


