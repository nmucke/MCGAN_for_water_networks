import torch
import networkx as nx
import pdb
from networkx.convert_matrix import from_numpy_matrix
import numpy as np
from networkx.convert_matrix import to_numpy_matrix
import copy

def get_adjacency_matrix(G):
    num_nodes = len(G.nodes)

    nodelist = [str(i+1) for i in range(num_nodes)]
    A = to_numpy_matrix(nx.Graph(G), nodelist=nodelist)
    A[A > 0] = 1

    return A

def create_graph_from_data(data, node_dict, edge_dict, G_old):

    G = copy.deepcopy(G_old)
    num_nodes = len(node_dict.keys())
    num_edges = len(edge_dict.keys())

    node_values = {}
    for key in node_dict.keys():
        node_values[node_dict[key]] = data[key].item()
    nx.set_node_attributes(G, node_values, "weight")

    edge_values = {}
    for key in edge_dict.keys():
        edge_values[edge_dict[key]] = data[num_nodes + key].item()
    nx.set_edge_attributes(G, edge_values, "weight")

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


