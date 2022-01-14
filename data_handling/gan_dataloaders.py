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


    def transform_state(self, data):
        return self.transformer.min_max_transform(data)

    def inverse_transform_state(self, data):
        return self.transformer.min_max_inverse_transform(data)

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):

        data_dict = nx.read_gpickle(self.data_path_state + str(idx))

        G = data_dict['graph']

        if self.transformer is not None:
            data = get_graph_data(G=G,
                                      transform=self.transform_state,
                                      separate_features=False)
        else:
            data = get_graph_data(G=G,
                                      transform=self.transformer,
                                      separate_features=False)
        if 'leak_pipe' in data_dict:
            par = torch.zeros([33])
            par[data_dict['leak_pipe']-2] = 1
            data = torch.cat([data, par])

        return data

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

















