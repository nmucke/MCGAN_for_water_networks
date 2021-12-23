import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import networkx as nx
from utils.graph_utils import get_graph_features

num_train = 100
# data_path_state = 'training_data_with_leak/network_'
data_path_state = '../data/training_data_no_leak/network_'
prior = np.zeros((num_train, 66))
for i in range(num_train):
    # data_dict = nx.read_gpickle(data_path_state + str(i))
    # G = data_dict['graph']
    G = nx.read_gpickle(data_path_state + str(i))

    data = get_graph_features(G=G, separate_features=False)

    prior[i, :] = data
    if i % 1000 == 0:
        print(i)
np.save('../prior_data_no_leak', prior)