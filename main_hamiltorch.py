import pdb
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
from utils.compute_statistics import get_statistics_from_latent_samples, get_demand_statistics
from plotting import plot_results
import numpy as np
import matplotlib.pyplot as plt
from utils.observation import observation_operator, get_test_observations
import wntr
from utils.graph_utils import get_incidence_mat, incidence_to_adjacency

torch.set_default_dtype(torch.float32)

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

    data_with_leak = True
    mix_leak = False
    gan_with_leak = True
    small_leak = False
    small_demand_variance_data = False
    small_demand_variance_gan = False

    leak_location_error = []
    std_list = []
    obs_error_list = []
    cases = range(0,10)
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
        activation = nn.LeakyReLU()
        transformer = None#transform_data(a=-1, b=1,
                          #           leak=gan_with_leak,
                          #           small=small_leak)

        generator_params = {'latent_dim': latent_dim,
                            'par_dim': 35,
                            'output_dim': 66,
                            'activation': activation,
                            'n_neurons': [16, 32, 48, 64],
                            'leak': gan_with_leak}

        generator = GAN_models.Generator(**generator_params).to(device)
        load_checkpoint(load_string, generator)
        generator.eval()

        true_data_dict = nx.read_gpickle(data_path)
        G_true = true_data_dict['graph']
        true_flow_rate = torch.tensor(true_data_dict['flow_rate'].values,
                                      dtype=torch.get_default_dtype(),
                                      device=device)
        true_head = torch.tensor(true_data_dict['head'].values,
                                      dtype=torch.get_default_dtype(),
                                      device=device)

        edge_obs_idx = range(0,34)
        node_obs_idx = range(0,32)
        obs_idx = {'edge_obs_idx': edge_obs_idx,
                   'node_obs_idx': node_obs_idx}

        edge_obs_std = 0.15
        node_obs_std = 0.15
        std = {'edge_obs_std': edge_obs_std,
               'node_obs_std': node_obs_std}

        obs_operator = lambda obs: observation_operator(data=obs,
                                                        obs_idx=obs_idx,
                                                        variable='both')
        flow_rate_obs_operator = lambda obs: observation_operator(data=obs,
                                                        obs_idx=edge_obs_idx,
                                                        variable='flow_rate')
        head_obs_operator = lambda obs: observation_operator(data=obs,
                                                        obs_idx=node_obs_idx,
                                                        variable='head')

        full_observations, noise_distribution = get_test_observations(data_dict=true_data_dict,
                                                  obs_idx=obs_idx,
                                                  std=std,
                                                  split=False,
                                                  device=device)

        z_init = torch.randn(1, latent_dim, requires_grad=True, device=device)
        z_map = compute_MAP(z=z_init,
                            observations=full_observations,
                            generator=generator,
                            obs_operator=obs_operator,
                            num_iter=2500)

        map_obs_error = {}
        map_full_error = {}

        map_obs_error['flow_rate'] = \
            torch.linalg.norm(full_observations[:, 0:len(edge_obs_idx)] \
          - obs_operator(generator(z_map))[:, 0:len(edge_obs_idx)]) \
            / torch.linalg.norm(full_observations[:, 0:len(edge_obs_idx)])
        map_obs_error['head'] = \
            torch.linalg.norm(full_observations[:, -len(node_obs_idx):] \
          - obs_operator(generator(z_map))[:, -len(node_obs_idx):]) \
            / torch.linalg.norm(full_observations[:, -len(node_obs_idx):])

        map_full_error['flow_rate'] = torch.linalg.norm(true_flow_rate-generator(z_map)[:, 0:34]) \
                                / torch.linalg.norm(true_flow_rate)
        map_full_error['head'] = torch.linalg.norm(true_head-generator(z_map)[:, 34:66]) \
                                / torch.linalg.norm(true_head)
        print(f'MAP OBSERVATION ERROR')
        print(f'Flow rate: {map_obs_error["flow_rate"].item():0.3f}')
        print(f'Head: {map_obs_error["head"].item():0.3f}')
        print(f'MAP FULL ERROR')
        print(f'Flow rate: {map_full_error["flow_rate"].item():0.3f}')
        print(f'Head: {map_full_error["head"].item():0.3f}')

        posterior_params = {'generator': generator,
                            'obs_operator': obs_operator,
                            'observations': full_observations,
                            'prior_mean': torch.zeros(latent_dim, device=device),
                            'prior_std': torch.ones(latent_dim, device=device),
                            'noise_mean': torch.cat([noise_distribution['flow_rate_noise_mean'],
                                                     noise_distribution['head_noise_mean']],
                                                     dim=1),
                            'noise_std': torch.cat([noise_distribution['flow_rate_noise_std'],
                                                    noise_distribution['head_noise_std']],
                                                    dim=1)}
        HMC_params = {'num_samples': 10000,
                      'step_size': 1.,
                      'num_steps_per_sample': 5,
                      'burn': 7500,
                      'integrator': hamiltorch.Integrator.IMPLICIT,
                      'sampler': hamiltorch.Sampler.HMC_NUTS,
                      'desired_accept_rate': 0.3}

        z_samples = hamiltonian_MC(z_init=torch.squeeze(z_map),
                                   posterior_params=posterior_params,
                                   HMC_params=HMC_params)

        MCGAN_results = \
            get_statistics_from_latent_samples(z_samples=z_samples,
                                               generator=generator,
                                               transform=transformer)

        flow_rate_std = torch.mean(MCGAN_results['flow_rate']['std']).item()
        head_std = torch.mean(MCGAN_results['head']['std']).item()
        std_list.append(0.5*(flow_rate_std + head_std))

        MCMC_obs_error = {}
        MCMC_full_error = {}

        MCMC_obs_error['flow_rate'] = \
              torch.linalg.norm(full_observations[:, 0:len(edge_obs_idx)] \
            - flow_rate_obs_operator(MCGAN_results['flow_rate']['mean'].unsqueeze(dim=0))) \
            / torch.linalg.norm(full_observations[:, 0:len(edge_obs_idx)])
        MCMC_obs_error['head'] = \
              torch.linalg.norm(full_observations[:, -len(node_obs_idx):] \
            - head_obs_operator(MCGAN_results['head']['mean'].unsqueeze(dim=0))) \
            / torch.linalg.norm(full_observations[:, -len(node_obs_idx):])

        MCMC_full_error['flow_rate'] = torch.linalg.norm(true_flow_rate-\
                                       MCGAN_results['flow_rate']['mean'].unsqueeze(dim=0)) \
                                     / torch.linalg.norm(true_flow_rate)
        MCMC_full_error['head'] = torch.linalg.norm(true_head-\
                                  MCGAN_results['head']['mean'].unsqueeze(dim=0)) \
                                / torch.linalg.norm(true_head)

        obs_error_list.append(0.5*(MCMC_obs_error['flow_rate'] + MCMC_obs_error['head']))

        print(f'MCMC OBSERVATION ERROR')
        print(f'Flow rate: {MCMC_obs_error["flow_rate"].item():0.3f}')
        print(f'Head: {MCMC_obs_error["head"].item():0.3f}')
        print(f'MCMC FULL ERROR')
        print(f'Flow rate: {MCMC_full_error["flow_rate"].item():0.3f}')
        print(f'Head: {MCMC_full_error["head"].item():0.3f}')

        MCGAN_results['demand'] = get_demand_statistics(data=MCGAN_results,
                                                        incidence_mat=incidence_mat)

        G_MCGAN = create_graph_from_data(MCGAN_results, G_true, incidence_mat)

        plt.figure()
        plt.plot(MCGAN_results['demand']['mean'].detach())
        plt.plot(true_data_dict['demand'].values[0])
        plt.show()

        if gan_with_leak:
            plot_results.plot_graph_results(G=G_MCGAN,
                                            G_true=G_true,
                                            leak={'true_leak': true_data_dict['leak']['pipe'],
                                                  'MCGAN_leak': MCGAN_results['leak']},
                                            save_string="MCGAN_results")
        else:
            plot_results.plot_graph_results(G=G_MCGAN,
                                            G_true=G_true,
                                            save_string="MCGAN_results")

        if gan_with_leak:
            print(f'Case {case} is done, leak: {leak_location_error[-1]},'
                  f' observation error: {obs_error_list[-1]:0.4f}, std: {std:0.4f}')
        else:
            print(f'Case {case} is done, leak present: {data_with_leak},'
                  f' observation error: {obs_error_list[-1]:0.4f}, std: {std:0.4f}')

    '''
    num_corrects = np.sum(leak_location_error)
    print(f'Total number of correct leaks: {num_corrects}')
    plt.figure()
    plt.subplot(1,2,1)
    if gan_with_leak:
        true_std_list = [i for (i, v) in zip(std_list, leak_location_error) if v]
        false_std_list = [i for (i, v) in zip(std_list, leak_location_error) if not v]
        plt.hist(true_std_list, label='True', density=False)
        plt.hist(false_std_list, alpha=0.7, label='False', density=False)
    elif mix_leak:
        plt.hist(std_list[0:int(num_cases/2)], label='Leak', density=False)
        plt.hist(std_list[-int(num_cases/2):], alpha=0.7, label='No Leak', density=False)
    plt.title('Standard Deviation')
    plt.legend()

    plt.subplot(1,2,2)
    if gan_with_leak:
        true_obs_error_list = [i for (i, v) in zip(obs_error_list, leak_location_error) if v]
        false_obs_error_list = [i for (i, v) in zip(obs_error_list, leak_location_error) if not v]
        plt.hist(true_obs_error_list, label='True', density=False)
        plt.hist(false_obs_error_list, alpha=0.7, label='False', density=False)
    elif mix_leak:
        plt.hist(obs_error_list[0:int(num_cases/2)], label='Leak', density=False, bins=30)
        plt.hist(obs_error_list[-int(num_cases/2):], alpha=0.7, label='No Leak', density=False, bins=30)
    plt.title('Observation Error')
    plt.legend()
    plt.show()
    '''






