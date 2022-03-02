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

        self.leak = leak

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
                self.max_flow_rate = max_min_vec.item()['max_flow_rate'] \
                               + 0.1*max_min_vec.item()['max_flow_rate']
                self.min_flow_rate = max_min_vec.item()['min_flow_rate'] \
                               - 0.1*max_min_vec.item()['min_flow_rate']
                self.max_head = max_min_vec.item()['max_head'] \
                               + 0.1*max_min_vec.item()['max_head']
                self.min_head = max_min_vec.item()['min_head'] \
                               - 0.1*max_min_vec.item()['min_head']
                self.max_leak_demand = max_min_vec.item()['max_leak_demand'] \
                               + 0.1*max_min_vec.item()['max_leak_demand']
                self.min_leak_demand = max_min_vec.item()['min_leak_demand'] \
                               - 0.1*max_min_vec.item()['min_leak_demand']

        else:
            max_min_vec = np.load('transforms/max_min_data_no_leak.npy',
                                  allow_pickle=True)
            self.max_flow_rate = max_min_vec.item()['max_flow_rate'] \
                                 + 0.1 * max_min_vec.item()['max_flow_rate']
            self.min_flow_rate = max_min_vec.item()['min_flow_rate'] \
                                 - 0.1 * max_min_vec.item()['min_flow_rate']
            self.max_head = max_min_vec.item()['max_head'] \
                            + 0.1 * max_min_vec.item()['max_head']
            self.min_head = max_min_vec.item()['min_head'] \
                            - 0.1 * max_min_vec.item()['min_head']

        self.max_vec_state = np.concatenate((self.max_flow_rate, self.max_head),
                                            axis=0)
        self.min_vec_state = np.concatenate((self.min_flow_rate, self.min_head),
                                            axis=0)
        self.max_vec_state = torch.tensor(self.max_vec_state, dtype=torch.get_default_dtype())
        self.min_vec_state = torch.tensor(self.min_vec_state, dtype=torch.get_default_dtype())

        if leak:
            self.max_vec_leak_demand = torch.tensor(self.max_leak_demand,
                                                    dtype=torch.get_default_dtype())
            self.min_vec_leak_demand = torch.tensor(self.min_leak_demand,
                                                    dtype=torch.get_default_dtype())

    def min_max_transform(self, data):

        if len(data.shape) == 1:
            data[0:66] = self.a + (data[0:66] - self.min_vec_state) * (self.b - self.a) \
                            / (self.max_vec_state - self.min_vec_state)
            if self.leak:
                data[66] = self.a + (data[66] - self.min_vec_leak_demand) * (self.b - self.a) \
                            / (self.max_vec_leak_demand - self.min_vec_leak_demand)
        else:
            data[:, 0:66] = self.a + (data[:, 0:66] - self.min_vec_state) * (self.b - self.a) \
                            / (self.max_vec_state - self.min_vec_state)
            if self.leak:
                data[:, 66] = self.a + (data[:, 66] - self.min_vec_leak_demand) * (self.b - self.a) \
                            / (self.max_vec_leak_demand - self.min_vec_leak_demand)

        return data

    def min_max_inverse_transform(self, data):
        if len(data.shape) == 1:
            data[0:66] = (data[0:66]-self.a)*(self.max_vec_state-self.min_vec_state) \
                   /(self.b-self.a) + self.min_vec_state
            if self.leak:
                data[66] = (data[66]-self.a)*(self.max_vec_leak_demand-self.min_vec_leak_demand) \
                   /(self.b-self.a) + self.min_vec_leak_demand
        else:
            data[:, 0:66] = (data[:, 0:66]-self.a)*(self.max_vec_state-self.min_vec_state) \
                   /(self.b-self.a) + self.min_vec_state
            if self.leak:
                data[:, 66] = (data[:, 66]-self.a)*(self.max_vec_leak_demand-self.min_vec_leak_demand) \
                   /(self.b-self.a) + self.min_vec_leak_demand

        return data

if __name__ == "__main__":
    data_path_state = '../data/training_data_no_leak/network_'

    flow_rate_list = []
    head_list = []
    leak_demand_list = []
    for i in range(200000):
        #G = nx.read_gpickle(data_path_state + str(i))
        data_dict = nx.read_gpickle(data_path_state + str(i))

        flow_rate = torch.tensor(data_dict['flow_rate'].values)[0]
        head = torch.tensor(data_dict['head'].values)[0]
        #leak_demand = torch.tensor(data_dict['leak']['demand'])

        flow_rate_list.append(flow_rate)
        head_list.append(head)
        #leak_demand_list.append(leak_demand)

        if i % 1000 == 0:
            print(i)
    flow_rate_list = torch.stack(flow_rate_list).detach().numpy()
    head_list = torch.stack(head_list).detach().numpy()
    #leak_demand_list = torch.stack(leak_demand_list).detach().numpy()

    max_flow_rate = np.max(flow_rate_list, axis=0)
    min_flow_rate = np.min(flow_rate_list, axis=0)
    max_head = np.max(head_list, axis=0)
    min_head = np.min(head_list, axis=0)
    #max_leak_demand = np.max(leak_demand_list)
    #min_leak_demand = np.min(leak_demand_list)

    np.save('max_min_data_no_leak', {'max_flow_rate': max_flow_rate,
                                       'min_flow_rate': min_flow_rate,
                                       'max_head': max_head,
                                       'min_head': min_head})
                                       #'max_leak_demand': max_leak_demand,
                                       #'min_leak_demand': min_leak_demand})





