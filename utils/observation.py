import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb


def observation_operator(data, obs_idx, variable='both'):
    if variable == 'head':
        obs = data[:, obs_idx]
    elif variable == 'flow_rate':
        obs = data[:, obs_idx]
    elif variable == 'both':
        edge_obs_idx = obs_idx['edge_obs_idx']
        node_obs_idx = obs_idx['node_obs_idx']
        edge_obs = data[:, edge_obs_idx]
        node_obs = data[:,34:66][:, node_obs_idx]
        obs = torch.cat([edge_obs, node_obs], dim=1)
    return obs

def add_noise_to_data(obs, noise_mean, noise_std):
    obs_noise = torch.normal(mean=noise_mean,
                             std=noise_std)
    obs += obs_noise
    return obs

def get_test_observations(data_dict, obs_idx, std, split=False, device='cpu'):
    flow_rate_observations = observation_operator(
        torch.tensor(data_dict['flow_rate'].values, dtype=torch.get_default_dtype()),
        obs_idx=obs_idx['edge_obs_idx'],
        variable='flow_rate').to(device)
    head_observations = observation_operator(
        torch.tensor(data_dict['head'].values, dtype=torch.get_default_dtype()),
        obs_idx=obs_idx['node_obs_idx'],
        variable='head').to(device)

    flow_rate_noise_mean = torch.zeros(flow_rate_observations.shape,
                                       device=device)
    head_noise_mean = torch.zeros(head_observations.shape,
                                  device=device)

    flow_rate_noise_std = std['edge_obs_std'] * torch.ones(flow_rate_observations.shape,
                                                dtype = torch.get_default_dtype(),
                                                device=device)
    head_noise_std = std['node_obs_std'] * torch.ones(head_observations.shape,
                                               dtype=torch.get_default_dtype(),
                                               device=device)

    flow_rate_observations = add_noise_to_data(flow_rate_observations,
                                               flow_rate_noise_mean,
                                               flow_rate_noise_std)
    head_observations = add_noise_to_data(head_observations,
                                          head_noise_mean,
                                          head_noise_std)

    noise_distribution = {'flow_rate_noise_mean': flow_rate_noise_mean,
                          'head_noise_mean': head_noise_mean,
                          'flow_rate_noise_std': flow_rate_noise_std,
                          'head_noise_std': head_noise_std}
    if split:
        return flow_rate_observations, head_observations, noise_distribution
    else:

        full_observations = torch.cat([flow_rate_observations,
                                       head_observations],
                                      dim=1)
        return full_observations, noise_distribution

