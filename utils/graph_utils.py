import torch
import networkx as nx
import pdb

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


