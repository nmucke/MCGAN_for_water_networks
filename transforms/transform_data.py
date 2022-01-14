import pdb
import numpy as np
import networkx as nx
import torch
from utils.graph_utils import get_graph_data


class transform_data():
    def __init__(self, a=-1, b=1, leak=True, small=False):

        super(transform_data, self).__init__()
        self.a = a
        self.b = b

        if leak:
            if small:
                max_min_vec = np.load('transforms/max_min_data_with_leak_small.npy',
                                      allow_pickle=True)
                self.max_vec = max_min_vec.item()['max_vec'] \
                               + 0.1*max_min_vec.item()['max_vec']
                self.min_vec = max_min_vec.item()['min_vec'] \
                               - 0.1*max_min_vec.item()['min_vec']
            else:
                max_min_vec = np.load('transforms/max_min_data_with_leak.npy',
                                      allow_pickle=True)
                self.max_vec = max_min_vec.item()['max_vec'] \
                               + 0.1*max_min_vec.item()['max_vec']
                self.min_vec = max_min_vec.item()['min_vec'] \
                               - 0.1*max_min_vec.item()['min_vec']
        else:
            max_min_vec = np.load('transforms/max_min_data.npy',
                                  allow_pickle=True)
            self.max_vec = max_min_vec.item()['max_vec']
            self.min_vec = max_min_vec.item()['min_vec']

        self.max_vec = torch.tensor(self.max_vec, dtype=torch.get_default_dtype())
        self.min_vec = torch.tensor(self.min_vec, dtype=torch.get_default_dtype())

    def min_max_transform(self, data):
        return self.a + (data-self.min_vec)*(self.b-self.a) \
               /(self.max_vec-self.min_vec)

    def min_max_inverse_transform(self, data):
        return (data-self.a)*(self.max_vec-self.min_vec) \
               /(self.b-self.a) + self.min_vec

if __name__ == "__main__":
    data_path_state = '../data/training_data_with_leak_small/network_'

    data = []
    for i in range(100000):
        #G = nx.read_gpickle(data_path_state + str(i))
        data_dict = nx.read_gpickle(data_path_state + str(i))
        G = data_dict['graph']
        graph_data = get_graph_data(G=G,
                              transform=None,
                              separate_features=False)

        data.append(graph_data)

        if i % 1000 == 0:
            print(i)

    data = torch.stack(data).detach().numpy()
    max_vec = np.max(data, axis=0)
    min_vec = np.min(data, axis=0)

    np.save('max_min_data_with_leak_small', {'max_vec':max_vec,
                             'min_vec':min_vec})





