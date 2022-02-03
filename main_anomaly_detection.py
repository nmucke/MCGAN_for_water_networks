import pdb
import numpy as npx
import torch.nn as nn
import torch
import models.GAN_models as GAN_models
from utils.load_checkpoint import load_checkpoint
from transforms.transform_data import transform_data
from utils.seed_everything import seed_everything
import networkx as nx
from utils.graph_utils import get_graph_data, get_adjacency_matrix, create_graph_from_data
from inference.maximum_a_posteriori import compute_MAP
from inference.MCMC import hamiltonian_MC
import hamiltorch
from utils.compute_statistics import get_statistics_from_latent_samples
from plotting import plot_results
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

def observation_operator(data, obs_idx):
    obs = data[obs_idx]
    return obs

def add_noise_to_data(obs, noise_mean, noise_std):
    obs_noise = torch.normal(mean=noise_mean,
                             std=noise_std)
    obs += obs_noise
    return obs

if __name__ == "__main__":

    seed_everything()

    cuda = False
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Running on {device}')


    data_with_leak = True
    mix_leak = True
    gan_with_leak = True
    small_leak = False
    small_demand_variance_data = True
    small_demand_variance_gan = True


    leak_location_error = []
    std_list = []
    obs_error_list = []
    crtic_score_list_leak = []
    crtic_score_list_no_leak = []
    alpha_score = []

    if gan_with_leak:
        if small_leak:
            load_string = 'model_weights/GAN_small_leak'
        else:
            if small_demand_variance_gan:
                load_string = 'model_weights/GAN_leak_small_demand_variance'
            else:
                load_string = 'model_weights/GAN_leak'
    else:
        if small_demand_variance_gan:
            load_string = 'model_weights/GAN_no_leak_small_demand_variance'
        else:
            load_string = 'model_weights/GAN_no_leak'

    latent_dim = 16
    activation = nn.Tanh()
    transformer = transform_data(a=-1, b=1,
                                 leak=gan_with_leak,
                                 small=small_leak)

    generator_params = {'latent_dim': latent_dim,
                        'par_dim': 33,
                        'output_dim': 66,
                        'activation': activation,
                        'n_neurons': [16, 32, 48, 64],
                        'leak': gan_with_leak}
    critic_neurons = [96, 80, 64, 48, 32, 16]
    critic_params_leak = {'activation': activation,
                          'n_neurons': critic_neurons}
    critic_params_no_leak = {'activation': activation,
                             'n_neurons': critic_neurons}

    critic_params_leak['input_dim'] = generator_params['output_dim'] + \
                                      generator_params['par_dim']

    critic_params_no_leak['input_dim'] = generator_params['output_dim']

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

    critic_leak = GAN_models.Critic(**critic_params_leak).to(device)
    load_checkpoint(critic_load_string_leak, critic=critic_leak)
    critic_leak.eval()

    critic_no = GAN_models.Critic(**critic_params_no_leak).to(device)
    load_checkpoint(critic_load_string_no_leak, critic=critic_no)
    critic_no.eval()

    cases = range(0,20)
    num_cases = len(cases)
    for case in cases:

        if mix_leak:
            if case < int(num_cases/2):
                data_with_leak = True
            else:
                data_with_leak = False

        if data_with_leak:
            if small_leak:
                data_path = 'data/test_data_with_leak_small/network_' + str(case)
            else:
                if small_demand_variance_data:
                    data_path = 'data/test_data_with_leak_small_demand_variance/network_' + str(case)
                else:
                    data_path = 'data/test_data_with_leak/network_' + str(case)
        else:
            if small_demand_variance_data:
                data_path = 'data/test_data_no_leak_small_demand_variance/network_' + str(case)
            else:
                data_path = 'data/test_data_no_leak/network_' + str(case)

        data_dict = nx.read_gpickle(data_path)
        G_true = data_dict['graph']
        adjacency_matrix = get_adjacency_matrix(G_true)
        node_positions = nx.get_node_attributes(G_true, 'pos')

        node_data_true, edge_data_true = get_graph_data(G=G_true,
                                                        transform=transformer.min_max_transform,
                                                        separate_features=True)
        data_true, node_dict, edge_dict = get_graph_data(G=G_true,
                                                         transform=transformer.min_max_transform,
                                                         separate_features=False,
                                                         get_dicts=True)
        node_data_true, edge_data_true, data_true = \
            node_data_true.to(device), edge_data_true.to(device), data_true.to(device)

        if data_with_leak:
            leak_pipe = data_dict['leak_pipe']
            leak_area = data_dict['leak_area']

        obs_idx = range(0, 32, 1)
        obs_std = 0.15

        obs_operator = lambda obs: observation_operator(obs, obs_idx)
        observations = obs_operator(data_true).to(device)

        noise_mean = torch.zeros(observations.shape, device=device)
        noise_std = obs_std*torch.ones(observations.shape, device=device)
        observations = add_noise_to_data(observations,
                                         noise_mean,
                                         noise_std)

        z_init = torch.randn(1, latent_dim, requires_grad=True, device=device)
        z_map = compute_MAP(z=z_init,
                            observations=observations,
                            generator=generator,
                            obs_operator=obs_operator,
                            num_iter=1000)

        posterior_params = {'generator': generator,
                            'obs_operator': obs_operator,
                            'observations': observations,
                            'prior_mean': torch.zeros(latent_dim, device=device),
                            'prior_std': torch.ones(latent_dim, device=device),
                            'noise_mean': noise_mean,
                            'noise_std': noise_std}
        HMC_params = {'num_samples': 10000,
                      'step_size': 1.,
                      'num_steps_per_sample': 5,
                      'burn': 8000,
                      'integrator': hamiltorch.Integrator.IMPLICIT,
                      'sampler': hamiltorch.Sampler.HMC_NUTS,
                      'desired_accept_rate': 0.3}

        z_samples = hamiltonian_MC(z_init=torch.squeeze(z_map),
                                   posterior_params=posterior_params,
                                   HMC_params=HMC_params)

        data_true = get_graph_data(G=G_true,
                                   transform=None,
                                   separate_features=False)
        node_data_true, edge_data_true = get_graph_data(G=G_true,
                                                        transform=None,
                                                        separate_features=True)
        node_data_true = node_data_true.to(device)
        edge_data_true = edge_data_true.to(device)

        MCGAN_results = \
            get_statistics_from_latent_samples(z_samples=z_samples,
                                               generator=generator,
                                               separate_features=False,
                                               transform=transformer.min_max_inverse_transform)

        MCGAN_results_separate = \
            get_statistics_from_latent_samples(z_samples=z_samples,
                                               generator=generator,
                                               separate_features=True,
                                               transform=transformer.min_max_inverse_transform,
                                               gan_with_leak=gan_with_leak)
        if gan_with_leak:
            leak_location_error.append(MCGAN_results['gen_leak_pipe_estimate']
                                       == leak_pipe)
        MCGAN_results = \
            get_statistics_from_latent_samples(z_samples=z_samples,
                                               generator=generator,
                                               separate_features=False,
                                               transform=None)

        std = torch.mean(MCGAN_results_separate['gen_node_std']) \
                + torch.mean(MCGAN_results_separate['gen_edge_std'])
        std_list.append(0.5*std.item())

        obs_error = torch.linalg.norm(observations-obs_operator(MCGAN_results['gen_mean'])) \
                    / torch.linalg.norm(observations)
        obs_error_list.append(obs_error.item())

        if gan_with_leak:
            critic_input_leak = torch.cat([MCGAN_results['gen_mean'],
                                      MCGAN_results['gen_leak_pipe'].mean(dim=0)])
            critic_input_no_leak = MCGAN_results['gen_mean']
        else:
            critic_input_leak = torch.cat([MCGAN_results['gen_mean'],
                                           1/33*torch.ones(33)])
            critic_input_no_leak = MCGAN_results['gen_mean']

        critic_score_leak = critic_leak(critic_input_leak)
        crtic_score_list_leak.append(critic_score_leak.item())

        critic_score_no_leak = critic_no(critic_input_no_leak)
        crtic_score_list_no_leak.append(critic_score_no_leak.item())

        if gan_with_leak:
            print(f'Case {case} is done, leak: {leak_location_error[-1]}, observation error: {obs_error:0.4f}, std: {std:0.4f}')
        else:
            print(f'Case {case} is done, leak present: {data_with_leak}, observation error: {obs_error:0.4f}, std: {std:0.4f}')

    plt.figure(figsize=(10,10))
    plt.suptitle(f'GAN with leak: {gan_with_leak}')

    plt.subplot(2,2,1)
    plt.hist(crtic_score_list_leak[0:int(num_cases/2)], label='Leak', density=False)
    plt.hist(crtic_score_list_leak[-int(num_cases/2):], alpha=0.7, label='No Leak', density=False)
    plt.title('Critic leak Score')
    plt.legend()

    plt.subplot(2,2,2)
    plt.hist(crtic_score_list_no_leak[0:int(num_cases/2)], label='Leak', density=False)
    plt.hist(crtic_score_list_no_leak[-int(num_cases/2):], alpha=0.7, label='No Leak', density=False)
    plt.title('Critic NO leak Score')
    plt.legend()

    plt.subplot(2,2,3)
    plt.hist(std_list[0:int(num_cases/2)], label='Leak', density=False)
    plt.hist(std_list[-int(num_cases/2):], alpha=0.7, label='No Leak', density=False)
    plt.title('Standard deviation')
    plt.legend()

    plt.subplot(2,2,4)
    plt.hist(obs_error_list[0:int(num_cases/2)], label='Leak', density=False)
    plt.hist(obs_error_list[-int(num_cases/2):], alpha=0.7, label='No Leak', density=False)
    plt.title('Observation error')
    plt.legend()

    plt.show()
