import pdb
import torch
import networkx as nx
from utils.graph_utils import get_graph_features, get_graph_data



class NetworkDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, num_files=100000, transformer=None):

        self.data_path_state = data_path
        self.num_files = num_files
        self.transformer = transformer

        self.state_IDs = [i for i in range(self.num_files)]

        self.dtype = torch.get_default_dtype()

    def transform_data(self, data):
        return self.transformer.min_max_transform(data)

    def inverse_transform_data(self, data):
        return self.transformer.min_max_inverse_transform(data)

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):

        data_dict = nx.read_gpickle(self.data_path_state + str(idx))

        flow_rate = torch.tensor(data_dict['flow_rate'].values, dtype=self.dtype)[0]
        head = torch.tensor(data_dict['head'].values, dtype=self.dtype)[0]
        demand = torch.tensor(data_dict['demand'].values, dtype=self.dtype)[0]

        data = torch.cat([flow_rate, head], dim=0)

        if 'leak' in data_dict:
            pars = torch.zeros([35,], dtype=self.dtype)
            pars[0] = torch.tensor(data_dict['leak']['demand'], dtype=self.dtype)
            pars[data_dict['leak']['pipe']] = 1
            data = torch.cat([data, pars], dim=0)

        if self.transformer is not None:
            data = self.transform_data(data)

        return data, demand

def get_dataloader(data_path,
                    num_files=100000,
                    transformer=None,
                    batch_size=512,
                    shuffle=True,
                    num_workers=2,
                    drop_last=True
                    ):

    dataset = NetworkDataset(data_path=data_path,
                             num_files=num_files,
                             transformer=transformer)
    dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               drop_last=drop_last)

    return dataloader

















