import pdb

import numpy as np
import networkx as nx
from utils.graph_utils import get_graph_data

num_train = 100000
# data_path_state = 'training_data_with_leak/network_'
data_path_state = '../data/training_data_no_leak/network_'
prior = np.zeros((num_train, 66))
for i in range(num_train):
    G = nx.read_gpickle(data_path_state + str(i))
    G = G['graph']

    data = get_graph_data(G=G, separate_features=False)

    prior[i, :] = data
    if i % 1000 == 0:
        print(i)
np.save('../prior_data_no_leak', prior)