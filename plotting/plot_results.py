import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx as nx



def plot_graph_results(G, true_data, MCGAN_data, prior_data,
                       node_dict, edge_dict, save_string="MCGAN_results"):
    edge_width = 4

    gen_node_mean = MCGAN_data["gen_node_mean"].detach().numpy()
    gen_edge_mean = MCGAN_data["gen_edge_mean"].detach().numpy()
    gen_node_std = MCGAN_data["gen_node_std"].detach().numpy()
    gen_edge_std = MCGAN_data["gen_edge_std"].detach().numpy()

    node_dict_reverse = dict((v, k) for k, v in node_dict.items())
    edge_dict_reverse = dict((v, k) for k, v in edge_dict.items())

    gen_node_color = [gen_node_mean[node_dict_reverse[key]]
                      for key in G.nodes.keys()]
    gen_edge_color = [gen_edge_mean[edge_dict_reverse[key]]
                      for key in G.edges.keys()]


    true_node_color = [true_data['node_data'][node_dict_reverse[key]].item()
                      for key in G.nodes.keys()]
    true_edge_color = [true_data['edge_data'][edge_dict_reverse[key]].item()
                      for key in G.edges.keys()]

    node_cmap = plt.get_cmap('viridis')
    edge_cmap = plt.get_cmap('viridis')
    pos = nx.get_node_attributes(G, 'pos')

    vmin = np.min(true_data['node_data'].detach().numpy())
    vmax = np.max(true_data['node_data'].detach().numpy())
    edge_min = np.min(true_data['edge_data'].detach().numpy())
    edge_max = np.max(true_data['edge_data'].detach().numpy())
    edge_index_list = list(G.edges.keys())

    plt.figure(figsize=(25,20))
    plt.subplot(3,2,1)
    nx.draw_networkx_nodes(G=G, pos=pos, vmin=vmin, vmax=vmax,
                           node_color=gen_node_color,
                           cmap=node_cmap,
                           with_labels=True)
    nx.draw_networkx_edges(G=G, pos=pos, edge_vmin=edge_min, edge_vmax=edge_max,
                           edge_color=gen_edge_color,
                           cmap=edge_cmap,
                           width=edge_width)
    nx.draw_networkx_labels(G=G, pos=pos)

    if "gen_leak_pipe_estimate" in MCGAN_data.keys():
        nx.draw_networkx_edge_labels(G=G, pos=pos,
         edge_labels={(edge_dict[MCGAN_data["gen_leak_pipe_estimate"] - 1][0],
                       edge_dict[MCGAN_data["gen_leak_pipe_estimate"] - 1][1]): 'X'},
         font_color='red', font_size=20)

    sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=plt.Normalize(vmin=vmin,
                                                                  vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=edge_min,
                                                                  vmax=edge_max))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    plt.title('MCGAN')

    plt.subplot(3,2,2)
    nx.draw_networkx_nodes(G=G, pos=pos, vmin=vmin, vmax=vmax,
                           node_color=true_node_color,
                           cmap=node_cmap,
                           with_labels=True)
    nx.draw_networkx_edges(G=G, pos=pos, edge_vmin=edge_min, edge_vmax=edge_max,
                           edge_color=true_edge_color,
                           cmap=edge_cmap,
                           width=edge_width)
    nx.draw_networkx_labels(G=G, pos=pos)

    nx.draw_networkx_edge_labels(G=G, pos=pos,
                edge_labels={(edge_dict[true_data["leak_pipe"]-1][0],
                              edge_dict[true_data["leak_pipe"]-1][1]): 'X'},
                font_color='red', font_size=20)
    sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=plt.Normalize(vmin=vmin,
                                                                  vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=edge_min,
                                                                  vmax=edge_max))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    plt.title('True')

    diff_node_data = np.abs(true_data['node_data'].detach().numpy() - gen_node_mean) \
                     /np.abs(true_data['node_data'].detach().numpy())
    diff_edge_data = np.abs(true_data['edge_data'].detach().numpy() - gen_edge_mean) \
                     /np.abs(true_data['edge_data'].detach().numpy())

    diff_node_color = [diff_node_data[node_dict_reverse[key]].item()
                      for key in G.nodes.keys()]
    diff_edge_color = [diff_edge_data[edge_dict_reverse[key]].item()
                      for key in G.edges.keys()]

    vmin = np.min(diff_node_data)
    vmax = np.max(diff_node_data)
    edge_min = np.min(diff_edge_data)
    edge_max = np.max(diff_edge_data)
    plt.subplot(3,2,3)
    nx.draw_networkx_nodes(G=G, pos=pos, vmin=vmin, vmax=vmax,
                           node_color=diff_node_color,
                           cmap=node_cmap,
                           with_labels=True)
    nx.draw_networkx_edges(G=G, pos=pos, edge_vmin=edge_min, edge_vmax=edge_max,
                           edge_color=diff_edge_color,
                           cmap=edge_cmap,
                           width=edge_width)
    nx.draw_networkx_labels(G=G, pos=pos)

    sm = plt.cm.ScalarMappable(cmap=node_cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    sm = plt.cm.ScalarMappable(cmap=node_cmap,
                               norm=plt.Normalize(vmin=edge_min, vmax=edge_max))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    plt.title('Relative Absolute Error')


    gen_std_node_color = [MCGAN_data["gen_node_std"].detach().numpy()[node_dict_reverse[key]]
                      for key in G.nodes.keys()]
    gen_std_edge_color = [MCGAN_data["gen_edge_std"].detach().numpy()[edge_dict_reverse[key]]
                      for key in G.edges.keys()]


    vmin_std = np.min(MCGAN_data["gen_node_std"].detach().numpy())
    vmax_std = np.max(MCGAN_data["gen_node_std"].detach().numpy())
    edge_min_std = np.min(MCGAN_data["gen_edge_std"].detach().numpy())
    edge_max_std = np.max(MCGAN_data["gen_edge_std"].detach().numpy())

    plt.subplot(3,2,4)
    nx.draw_networkx_nodes(G=G, pos=pos, vmin=vmin_std, vmax=vmax_std,
                           node_color=gen_std_node_color,
                           cmap=node_cmap,
                           with_labels=True)
    nx.draw_networkx_edges(G=G, pos=pos, edge_vmin=edge_min_std, edge_vmax=edge_max_std,
                           edge_color=gen_std_edge_color,
                           cmap=edge_cmap,
                           width=edge_width)
    nx.draw_networkx_labels(G=G, pos=pos)
    sm = plt.cm.ScalarMappable(cmap=node_cmap,
                               norm=plt.Normalize(vmin=vmin_std,
                                                  vmax=vmax_std))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    sm = plt.cm.ScalarMappable(cmap=edge_cmap,
                               norm=plt.Normalize(vmin=edge_min_std,
                                                  vmax=edge_max_std))
    sm.set_array([])
    cbar = plt.colorbar(sm)

    plt.title('Standard Deviation')


    prior_mean = np.mean(prior_data, axis=0)
    diff_node_data = np.abs(prior_mean[0:32] - gen_node_mean)
    diff_edge_data = np.abs(prior_mean[-34:] - gen_edge_mean)

    diff_node_color = [diff_node_data[node_dict_reverse[key]].item()
                      for key in G.nodes.keys()]
    diff_edge_color = [diff_edge_data[edge_dict_reverse[key]].item()
                      for key in G.edges.keys()]

    vmin = np.min(diff_node_data)
    vmax = np.max(diff_node_data)
    edge_min = np.min(diff_edge_data)
    edge_max = np.max(diff_edge_data)
    plt.subplot(3,2,5)
    nx.draw_networkx_nodes(G=G, pos=pos, vmin=vmin, vmax=vmax,
                           node_color=diff_node_color,
                           cmap=node_cmap,
                           with_labels=True)
    nx.draw_networkx_edges(G=G, pos=pos, edge_vmin=edge_min, edge_vmax=edge_max,
                           edge_color=diff_edge_color,
                           cmap=edge_cmap,
                           width=edge_width)
    nx.draw_networkx_labels(G=G, pos=pos)

    sm = plt.cm.ScalarMappable(cmap=node_cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    sm = plt.cm.ScalarMappable(cmap=node_cmap,
                               norm=plt.Normalize(vmin=edge_min, vmax=edge_max))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    plt.title('Prior Error')

    plt.savefig(save_string+'.pdf')
    plt.show()


def plot_histogram_results(G, prior_data, true_data, MCGAN_data,
                           node_dict, edge_dict,
                           save_string="MCGAN_histogram_results"):

    plt.figure(figsize=(15, 15))
    for i in range(MCGAN_data["gen_node_samples"].shape[1]):
        plt.subplot(6, 6, i + 1)
        plt.hist(MCGAN_data["gen_node_samples"][:, i].detach().numpy(),
                 bins=50, density=True, label='Generator')
        plt.hist(prior_data["node_data"][:, i], bins=50, label='Prior',
                 alpha=0.8, density=True)
        plt.axvline(x=true_data['node_data'][i], ymin=0, ymax=1, color='k',
                    linewidth=3.)
        plt.title(node_dict[i])
    plt.tight_layout(pad=2.0)
    plt.savefig('distributions_node.pdf')
    plt.show()

    plt.figure(figsize=(15, 15))
    for i in range(MCGAN_data["gen_edge_samples"].shape[1]):
        plt.subplot(6, 6, i + 1)
        plt.hist(MCGAN_data["gen_edge_samples"][:, i].detach().numpy(),
                 bins=50, density=True, label='Generator')
        plt.hist(prior_data["edge_data"][:, i], bins=50, label='Prior',
                 alpha=0.8, density=True)
        plt.axvline(x=true_data['edge_data'][i], ymin=0, ymax=1,
                    color='k', linewidth=3.)
        plt.title(edge_dict[i])

    plt.tight_layout(pad=2.0)
    plt.savefig(save_string + '.pdf')
    plt.show()

def plot_leak_location(gen_leak_location, true_leak_location):
    plt.figure()
    plt.bar(range(2, 35), torch.mean(gen_leak_location, dim=0).detach().numpy())
    plt.axvline(true_leak_location, color='k', linewidth=3.)
    plt.savefig('bar_plot.pdf')
    plt.show()