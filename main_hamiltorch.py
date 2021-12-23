import copy
import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from data_handling.gan_dataloaders import get_dataloader
import models.GAN_models as GAN_models
from utils.load_checkpoint import load_checkpoint
from transforms.transform_data import transform_data
from training.training_GAN import TrainGAN
from utils.seed_everything import seed_everything
import networkx as nx
from utils.graph_utils import get_graph_features
from inference.maximum_a_posteriori import compute_MAP
from inference.MCMC import hamiltonian_MC
import hamiltorch
from networkx.linalg.graphmatrix import adjacency_matrix, incidence_matrix
torch.set_default_dtype(torch.float64)
from utils.compute_statistics import get_statistics_from_latent_samples
from plotting import plot_results

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


    train_with_leak = True
    small_leak = False
    obs_idx = range(32)
    obs_std = .001

    case = 0
    if train_with_leak:
        if small_leak:
            data_path = 'data/training_data_with_leak_small/network_' + str(case)
        else:
            data_path = 'data/training_data_with_leak/network_' + str(case)
    else:
        data_path = 'data/training_data_no_leak/network_' + str(case)

    load_string = 'model_weights/GAN_leak'

    latent_dim = 32
    activation = nn.LeakyReLU()
    transformer = transform_data(a=-1, b=1,
                                 leak=train_with_leak,
                                 small=small_leak)

    dataloader_params = {'data_path': data_path,
                         'num_files': 100000,
                         'transformer': transformer,
                         'batch_size': 512,
                         'shuffle': True,
                         'num_workers': 8,
                         'drop_last': True}
    generator_params = {'latent_dim': latent_dim,
                        'par_dim': 33,
                        'output_dim': 66,
                        'activation': activation,
                        'n_neurons': [16, 32, 64],
                        'leak': train_with_leak}

    generator = GAN_models.Generator(**generator_params).to(device)
    load_checkpoint(load_string, generator)

    dataloader = get_dataloader(**dataloader_params)

    data_dict = nx.read_gpickle(data_path)
    G_true = data_dict['graph']
    node_data_true, edge_data_true = get_graph_features(G=G_true,
                                              transform=transformer.min_max_transform,
                                              separate_features=True)
    data_true = get_graph_features(G=G_true,
                                  transform=transformer.min_max_transform,
                                  separate_features=False)
    node_data_true, edge_data_true, data_true = \
    node_data_true.to(device), edge_data_true.to(device), data_true.to(device)

    leak_pipe = data_dict['leak_pipe']
    leak_area = data_dict['leak_area']

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
                        obs_operator=obs_operator)

    obs_error = torch.linalg.norm(observations-\
                  obs_operator(generator(z_map)[0])) \
                / torch.linalg.norm(observations)
    full_error = torch.linalg.norm(data_true-generator(z_map)[0,0:generator_params['output_dim']]) \
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
    HMC_params = {'num_samples': 100,
                  'step_size': 1.,
                  'num_steps_per_sample': 5,
                  'burn': 50,
                  'integrator': hamiltorch.Integrator.IMPLICIT}

    z_samples = hamiltonian_MC(z_init=torch.squeeze(z_map),
                               posterior_params=posterior_params,
                               HMC_params=HMC_params)

    MCGAN_results = \
        get_statistics_from_latent_samples(z_samples=z_samples,
                                           generator=generator,
                                           separate_features=False,
                                           transform=transformer.min_max_inverse_transform)

    MCGAN_results_separate = \
        get_statistics_from_latent_samples(z_samples=z_samples,
                                           generator=generator,
                                           separate_features=True,
                                           transform=transformer.min_max_inverse_transform)

    plot_results.plot_graph_results(G=G_true,
                                    true_data={"node_data": node_data_true,
                                               "edge_data": edge_data_true},
                                    MCGAN_data=MCGAN_results_separate)

    prior = np.load('prior_data_no_leak.npy')
    plot_results.plot_histogram_results(G=G_true,
                                        prior_data={"node_data": prior[:,0:32],
                                                    "edge_data": prior[:,32:]},
                                        true_data={"node_data": node_data_true,
                                                   "edge_data": edge_data_true},
                                        MCGAN_data=MCGAN_results_separate)
    pdb.set_trace()

'''
num_train = 100000
#data_path_state = 'training_data_with_leak/network_'
data_path_state = 'training_data/base_demand_'
prior = np.zeros((num_train,66))
for i in range(num_train):
    #data_dict = nx.read_gpickle(data_path_state + str(i))
    #G = data_dict['graph']
    G = nx.read_gpickle(data_path_state + str(i))
    node_head = nx.get_node_attributes(G, 'weight')
    edge_flowrate = nx.get_edge_attributes(G, 'weight')

    node_weigts = [node_head[key][0] for key in node_head.keys()]
    edge_weights = [edge_flowrate[key][0] for key in edge_flowrate.keys()]

    node_tensor = np.asarray(node_weigts)
    edge_tensor = np.asarray(edge_weights)

    data = np.concatenate((node_tensor, edge_tensor), axis=0)
    prior[i,:] = data
    if i % 1000 == 0:
        print(i)
np.save('prior_data_no_leak', prior)
'''



plt.figure()
plt.bar(range(2,35), torch.mean(gen_leak_pipe,dim=0).detach().numpy())
plt.axvline(leak_pipe, color='k', linewidth=3.)
plt.savefig('bar_plot.pdf')
plt.show()


prior = np.load('prior_data_no_leak.npy')

#transform_no_leak = transform_data.transform_data(leak=False)
#prior = transform_no_leak.min_max_transform(prior)
gen_samples = gen_samples.detach().numpy()
gen_samples = transform.min_max_inverse_transform(gen_samples)

lol = nx.get_node_attributes(G, 'weight')
plt.figure(figsize=(15,15))
for i in range(len(gen_node_data)):
    plt.subplot(6,6,i+1)
    plt.hist(gen_samples[:,i], bins=50, density=True, label='Generator')
    plt.hist(prior[:,i], bins=50, label='Prior', alpha=0.8, density=True)
    plt.axvline(x=true_node_data_no_transform[i],ymin=0,ymax=1, color='k', linewidth=3.)
    plt.title(list(G.nodes)[i])
    #plt.xlim([80,220])

plt.tight_layout(pad=2.0)
plt.savefig('distributions_node_with_leak.pdf')
plt.show()

lol = nx.get_edge_attributes(G, 'weight')
plt.figure(figsize=(15,15))
for i in range(32,32+len(gen_edge_data)-1):
    plt.subplot(6,6,i+1-32)
    plt.hist(gen_samples[:,i], bins=50, density=True, label='Generator')
    plt.hist(prior[:,i], bins=50, label='Prior', alpha=0.8, density=True)
    plt.axvline(x=true_edge_data_no_transform[i-32],ymin=0,ymax=1, color='k', linewidth=3.)
    plt.title(list(G.edges)[i-32])
    #plt.xlim([80,220])

plt.tight_layout(pad=2.0)
plt.savefig('distributions_edge_with_leak.pdf')
plt.show()

print(f'Leak location: {leak_pipe}')
print(f'Leak area: {leak_area}')
print(f'Var(leak location): {torch.var(gen_leak_pipe):0.3f}')






