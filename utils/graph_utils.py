import torch
import networkx as nx
import pdb
from networkx.convert_matrix import from_numpy_matrix
import numpy as np
from networkx.convert_matrix import to_numpy_matrix
import copy
import pandas as pd

def get_incidence_mat(WNTR_model):
    incidence_mat = np.zeros((34,32))

    for i in range(1,33):
        inlet_ids = WNTR_model.get_links_for_node(str(i), 'INLET')
        outlet_ids = WNTR_model.get_links_for_node(str(i), 'OUTLET')
        for j in inlet_ids:
            incidence_mat[int(j)-1, i-1] = -1
        for j in outlet_ids:
            incidence_mat[int(j)-1, i-1] = 1
    return incidence_mat

def incidence_to_adjacency(incidence_mat):
    adjacency_mat = (np.dot(np.abs(incidence_mat).T, np.abs(incidence_mat)) > 0).astype(int)
    np.fill_diagonal(adjacency_mat, 0)
    return adjacency_mat

def get_edge_lists():
    A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 10, 14, 15, 16, 17, 18, 19, 3,
         20, 21, 20, 23, 24, 25, 26, 27, 23, 28, 29, 30, 31, 32]
    B = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 3, 20,
         21, 22, 23, 24, 25, 26, 27, 16, 28, 29, 30, 31, 32, 25]
    return A, B

def get_adjacency_matrix(G):
    num_nodes = len(G.nodes)

    nodelist = [str(i+1) for i in range(num_nodes)]
    A = to_numpy_matrix(nx.Graph(G), nodelist=nodelist)
    A[A > 0] = 1

    return A




def create_graph_from_data(data, G_old, incidence_mat):
    flow_rate = data['flow_rate']['mean'].unsqueeze(dim=1).detach().numpy()
    head = data['head']['mean'].unsqueeze(dim=1).detach().numpy()
    demand = data['demand']['mean'].unsqueeze(dim=1).detach().numpy()

    flow_rate_std = data['flow_rate']['std'].unsqueeze(dim=1).detach().numpy()
    head_std = data['head']['std'].unsqueeze(dim=1).detach().numpy()
    demand_std = data['demand']['std'].unsqueeze(dim=1).detach().numpy()

    head_df = pd.DataFrame(data=head.T, columns=range(1, 33))
    demand_df = pd.DataFrame(data=demand.T, columns=range(1, 33))

    head_std_df = pd.DataFrame(data=head_std.T, columns=range(1, 33))
    demand_std_df = pd.DataFrame(data=demand_std.T, columns=range(1, 33))

    pos = {}
    for i in range(1, 33):
        pos[i] = nx.get_node_attributes(G_old, 'pos')[i]

    A, B = get_edge_lists()
    df = pd.DataFrame(flow_rate[:, 0:1], columns=['flow_rate'])
    df['flow_rate_std'] = flow_rate_std
    df['a'] = A
    df['b'] = B
    G = nx.from_pandas_edgelist(df, source='a', target='b',
                                edge_attr=['flow_rate', 'flow_rate_std'])
    nx.set_node_attributes(G, head_df, name='head')
    nx.set_node_attributes(G, head_std_df, name='head_std')
    nx.set_node_attributes(G, demand_df, name='demand')
    nx.set_node_attributes(G, demand_std_df, name='demand_std')
    nx.set_node_attributes(G, pos, name='pos')

    return G

def get_graph_data(G, transform=None, separate_features=True,
                   get_dicts=False):

    num_nodes = len(G.nodes)
    num_edges = len(G.edges)

    nodes = torch.zeros((num_nodes,))
    edges = torch.zeros((num_edges,))

    node_attributes = nx.get_node_attributes(G, 'weight')
    edge_attributes = nx.get_edge_attributes(G, 'weight')

    node_dict = {}
    for label_string in G.nodes.keys():
        label_int = int(label_string)
        nodes[label_int-1] = torch.tensor(node_attributes[label_string][0])
        node_dict[label_int-1] = label_string

    edge_dict = {}
    for label_string in G.edges.keys():
        label_int = int(label_string[2])
        edges[label_int-1] = torch.tensor(edge_attributes[label_string][0])
        edge_dict[label_int-1] = label_string

    data = torch.cat([nodes, edges], dim=0)

    if transform is not None:
        data = transform(data)

    if separate_features:
        node_data = data[0:32]
        edge_data = data[-34:]

        if get_dicts:
            return node_data, edge_data, node_dict, edge_dict
        else:
            return node_data, edge_data
    else:
        if get_dicts:
            return data, node_dict, edge_dict
        else:
            return data

def get_graph_features(G, transform=None, separate_features=True):

    node_head = nx.get_node_attributes(G, 'weight')
    edge_flowrate = nx.get_edge_attributes(G, 'weight')

    node_weights = torch.tensor([node_head[key][0] for key in node_head.keys()],
                               dtype=torch.get_default_dtype())
    edge_weights = torch.tensor([edge_flowrate[key][0] for key in edge_flowrate.keys()],
                                dtype=torch.get_default_dtype())

    data = torch.cat([node_weights, edge_weights], dim=0)

    if transform is not None:
        data = transform(data)

    if separate_features:
        node_data = data[0:32]
        edge_data = data[-34:]
        return node_data, edge_data
    else:
        return data


