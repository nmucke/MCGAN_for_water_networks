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
    gan_with_leak = False
    small_leak = False

    leak_location_error = []
    std_list = []
    obs_error_list = []
    num_cases = 100
    for case in range(0,num_cases):
        if mix_leak:
            if case < int(num_cases/2):
                data_with_leak = True
            else:
                data_with_leak = False

        if data_with_leak:
            if small_leak:
                data_path = 'data/training_data_with_leak_small/network_' + str(case)
            else:
                data_path = 'data/training_data_with_leak/network_' + str(case)
        else:
            data_path = 'data/training_data_no_leak/network_' + str(case)

        if gan_with_leak:
            if small_leak:
                load_string = 'model_weights/GAN_small_leak'
            else:
                load_string = 'model_weights/GAN_leak'
        else:
            load_string = 'model_weights/GAN_no_leak'

        latent_dim = 32
        activation = nn.LeakyReLU()
        transformer = transform_data(a=-1, b=1,
                                     leak=gan_with_leak,
                                     small=small_leak)

        generator_params = {'latent_dim': latent_dim,
                            'par_dim': 33,
                            'output_dim': 66,
                            'activation': activation,
                            'n_neurons': [32, 48, 64, 80, 96],
                            'leak': gan_with_leak}

        generator = GAN_models.Generator(**generator_params).to(device)
        load_checkpoint(load_string, generator)
        generator.eval()


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
        obs_std = 0.2

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
                            num_iter=1500)

        obs_error = torch.linalg.norm(observations- \
                                      obs_operator(generator(z_map)[0])) \
                    / torch.linalg.norm(observations)
        full_error = torch.linalg.norm(data_true-generator(z_map)
        [0,0:generator_params['output_dim']]) \
                     / torch.linalg.norm(data_true)
        print(f'Observation error: {obs_error:0.4f}')
        print(f'Full error: {full_error:0.4f}')

        posterior_params = {'generator': generator,
                            'obs_operator': obs_operator,
                            'observations': observations,
                            'prior_mean': torch.zeros(latent_dim, device=device),
                            'prior_std': torch.ones(latent_dim, device=device),
                            'noise_mean': noise_mean,
                            'noise_std': noise_std}
        HMC_params = {'num_samples': 2,
                      'step_size': 1.,
                      'num_steps_per_sample': 5,
                      'burn': 1,
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
        node_data_true, edge_data_true = node_data_true.to(device), edge_data_true.to(device)

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
        std_list.append(torch.sum(torch.abs(MCGAN_results['gen_std'])).item())

        #obs_error = torch.linalg.norm(observations-obs_operator(MCGAN_results['gen_mean'])) \
        #            / torch.linalg.norm(observations)
        #obs_error_list.append(obs_error.item())


        obs_error = torch.linalg.norm(observations-obs_operator(generator(z_map)[0])) \
                    / torch.linalg.norm(observations)
        obs_error_list.append(obs_error.item())

        print(f'Case {case} is done.')


    plt.figure()
    plt.subplot(1,2,1)
    if gan_with_leak:
        true_std_list = [i for (i, v) in zip(std_list, leak_location_error) if v]
        false_std_list = [i for (i, v) in zip(std_list, leak_location_error) if not v]
        plt.hist(true_std_list, label='True', density=True)
        plt.hist(false_std_list, alpha=0.7, label='False', density=True)
    elif mix_leak:
        plt.hist(std_list[0:int(num_cases/2)], label='Leak', density=True)
        plt.hist(std_list[-int(num_cases/2):], alpha=0.7, label='No Leak', density=True)
    plt.title('Standard Deviation')
    plt.legend()

    plt.subplot(1,2,2)
    if gan_with_leak:
        true_obs_error_list = [i for (i, v) in zip(obs_error_list, leak_location_error) if v]
        false_obs_error_list = [i for (i, v) in zip(obs_error_list, leak_location_error) if not v]
        plt.hist(true_obs_error_list, label='True', density=True)
        plt.hist(false_obs_error_list, alpha=0.7, label='False', density=True)
    elif mix_leak:
        plt.hist(obs_error_list[0:int(num_cases/2)], label='Leak', density=True)
        plt.hist(obs_error_list[-int(num_cases/2):], alpha=0.7, label='No Leak', density=True)
    plt.title('Observation Error')
    plt.legend()
    plt.show()

    prior = np.load('prior_data_no_leak.npy')
    G_true = create_graph_from_data(data_true, node_dict, edge_dict, G_true)
    plot_results.plot_graph_results(G=G_true,
                                    true_data={"node_data": node_data_true,
                                               "edge_data": edge_data_true,
                                               "leak_pipe": leak_pipe},
                                    MCGAN_data=MCGAN_results_separate,
                                    prior_data=prior,
                                    node_dict=node_dict,
                                    edge_dict=edge_dict)

    plot_results.plot_histogram_results(G=G_true,
                                        prior_data={"node_data": prior[:,0:32],
                                                    "edge_data": prior[:,32:]},
                                        true_data={"node_data": node_data_true,
                                                   "edge_data": edge_data_true},
                                        node_dict=node_dict,
                                        edge_dict=edge_dict,
                                        MCGAN_data=MCGAN_results_separate)
    if gan_with_leak:
        plot_results.plot_leak_location(gen_leak_location=MCGAN_results['gen_leak_pipe'],
                                        true_leak_location=leak_pipe)






