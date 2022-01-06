import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx as nx



def plot_graph_results(G, true_data, MCGAN_data, save_string="MCGAN_results"):

    node_cmap = plt.get_cmap('viridis')
    edge_cmap = plt.get_cmap('viridis')
    pos = nx.get_node_attributes(G, 'pos')

    vmin = np.min(true_data['node_data'].detach().numpy())
    vmax = np.max(true_data['node_data'].detach().numpy())
    edge_min = np.min(true_data['edge_data'].detach().numpy())
    edge_max = np.max(true_data['edge_data'].detach().numpy())

    plt.figure(figsize=(15,12))
    plt.subplot(2,2,1)
    nx.draw(G,pos,with_labels=True, arrows=True,
            vmin=vmin, vmax=vmax, width=MCGAN_data["gen_edge_mean"].detach().numpy(),
            edge_vmin=edge_min, edge_vmax=edge_max,
            edge_color=MCGAN_data["gen_edge_mean"].detach().numpy(), edge_cmap=edge_cmap,
            node_color=MCGAN_data["gen_node_mean"].detach().numpy(), cmap=node_cmap,)
    sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm)

    plt.subplot(2,2,2)
    nx.draw(G,pos,with_labels=True, arrows=True,
            vmin=vmin, vmax=vmax,
            edge_vmin=edge_min, edge_vmax=edge_max,
            node_color=true_data['node_data'].detach().numpy(), cmap=node_cmap,)
    sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm)


    diff_node_data = np.abs(true_data['node_data'].detach().numpy() \
                            - MCGAN_data["gen_node_mean"].detach().numpy()) \
                     /np.sum(np.abs(true_data['node_data'].detach().numpy()))
    diff_edge_data = np.abs(true_data['edge_data'].detach().numpy() \
                            - MCGAN_data["gen_edge_mean"].detach().numpy()) \
                     /np.sum(np.abs(true_data['edge_data'].detach().numpy()))

    vmin = np.min(diff_node_data)
    vmax = np.max(diff_node_data)
    edge_min = np.min(diff_edge_data)
    edge_max = np.max(diff_edge_data)
    plt.subplot(2,2,3)
    nx.draw(G,pos,with_labels=True, arrows=True,
            vmin=vmin, vmax=vmax,
            edge_vmin=edge_min, edge_vmax=edge_max,
            node_color=diff_node_data, cmap=node_cmap,)
    sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm)


    vmin_std = np.min(MCGAN_data["gen_node_std"].detach().numpy())
    vmax_std = np.max(MCGAN_data["gen_node_std"].detach().numpy())
    edge_min_std = np.min(MCGAN_data["gen_edge_std"].detach().numpy())
    edge_max_std = np.max(MCGAN_data["gen_edge_std"].detach().numpy())

    plt.subplot(2,2,4)
    nx.draw(G,pos,with_labels=True, arrows=True,
            vmin=vmin_std, vmax=vmax_std, width=50*MCGAN_data["gen_edge_std"].detach().numpy(),
            edge_vmin=edge_min_std, edge_vmax=edge_max_std,
            edge_color=MCGAN_data["gen_edge_std"].detach().numpy(), edge_cmap=edge_cmap,
            node_color=MCGAN_data["gen_node_std"].detach().numpy(), cmap=node_cmap,)
    sm = plt.cm.ScalarMappable(cmap=node_cmap,
                               norm=plt.Normalize(vmin=vmin_std,
                                                  vmax=vmax_std))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    sm = plt.cm.ScalarMappable(cmap=edge_cmap,
                               norm=plt.Normalize(vmin=edge_min,
                                                  vmax=edge_max))
    sm.set_array([])
    cbar = plt.colorbar(sm)

    plt.savefig(save_string+'.pdf')
    plt.show()

def plot_histogram_results(G, prior_data, true_data, MCGAN_data, 
                           save_string="MCGAN_histogram_results"):

    plt.figure(figsize=(15, 15))
    for i in range(MCGAN_data["gen_node_samples"].shape[1]):
        plt.subplot(6, 6, i + 1)
        plt.hist(MCGAN_data["gen_node_samples"][:, i].detach().numpy(),
                 bins=50, density=True, label='Generator')
        plt.hist(prior_data["node_data"][:, i], bins=50, label='Prior', alpha=0.8, density=True)
        plt.axvline(x=true_data['node_data'][i], ymin=0, ymax=1, color='k',
                    linewidth=3.)
        plt.title(list(G.nodes)[i])
        # plt.xlim([80,220])
    plt.tight_layout(pad=2.0)
    plt.savefig('distributions_node_with_leak.pdf')
    plt.show()

    plt.figure(figsize=(15, 15))
    for i in range(MCGAN_data["gen_edge_samples"].shape[1]):
        plt.subplot(6, 6, i + 1)
        plt.hist(MCGAN_data["gen_edge_samples"][:, i].detach().numpy(),
                 bins=50, density=True, label='Generator')
        plt.hist(prior_data["edge_data"][:, i], bins=50, label='Prior', alpha=0.8, density=True)
        plt.axvline(x=true_data['edge_data'][i], ymin=0, ymax=1,
                    color='k', linewidth=3.)
        # plt.title(list(G.edges))
        # plt.xlim([80,220])

    plt.tight_layout(pad=2.0)
    plt.savefig(save_string + '.pdf')
    plt.show()

def plot_leak_location(gen_leak_location, true_leak_location):
    plt.figure()
    plt.bar(range(2, 35), torch.mean(gen_leak_location, dim=0).detach().numpy())
    plt.axvline(true_leak_location, color='k', linewidth=3.)
    plt.savefig('bar_plot.pdf')
    plt.show()