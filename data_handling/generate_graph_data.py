
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import wntr
import networkx as nx
import copy
import pdb
import seaborn as sbn
import networkx as nx
from networkx.linalg.graphmatrix import adjacency_matrix
import ray


# Getting path for the input file
inputfiles_folder_name = '../Input_files_EPANET'
filename = 'Hanoi_base_demand.inp'
path_file = os.path.join(inputfiles_folder_name,filename)

# Reading the input file into EPANET
inp_file = path_file
wn = wntr.network.WaterNetworkModel(inp_file)
wn1 = wntr.network.WaterNetworkModel(inp_file)

# store no of nodes and links
num_nodes = len(wn1.node_name_list)
num_links = len(wn1.link_name_list)

# create array that contains the base reservoir head and demand at nodes
base_demands = np.zeros((num_nodes))
base_demands[0]= wn1.get_node(1).head_timeseries.base_value
for i in range(1,num_nodes):
    base_demands[i] = wn1.get_node(i+1).demand_timeseries_list[0].base_value

# define standard deviation matrix
std_dev = base_demands*0.2
std_dev[0] = base_demands[0]*0.05


# function to create a covariance matrix
def cov_mat_fixed(corr_demands, corr_reservoir_nodes):
    N = num_nodes

    mat_corr = np.zeros((N, N))  # initializing matrix

    mat = np.full((N - 1, N - 1),
                  corr_demands)  # step 1 for a symmetric matrix of n-1 by n-1
    mat_symm = (mat + mat.T) / 2  # step 2

    diag = np.ones(
        N)  # setting up the diagonal matrix, variance of nodal demands
    np.fill_diagonal(mat_symm, diag)
    mat_corr[1:, 1:] = mat_symm

    mat_corr[0, 0] = 1  # element (0,0) which is variance of resevoir head

    top = np.full((N - 1),
                  corr_reservoir_nodes)  # covariance between reservoir head
    # and nodal demands
    mat_corr[0, 1:] = top
    mat_corr[1:, 0] = top

    Diag = np.diag(std_dev)
    cov_mat = Diag * mat_corr * Diag

    return cov_mat

#Covmat for all experiments
covmat_base = cov_mat_fixed(0.6,0.0)

# Column names for the dataframe that will store results. Note that the Reservoir Demand and Head comes after all
# other nodes. Naming is done accordingly.

col_names=['Demand'+str(i) for i in range(2,33)]+\
          ['Demand_Reservoir']+\
          ['Node_head'+str(i) for i in range(2,33)]+\
          ['Res_Head']+\
          ['Link_flow'+str(i) for i in range(1,35)]



def graph_generation(cov_mat,nodes_data):

    #getting samples
    train_data_raw = np.random.multivariate_normal(nodes_data,cov_mat,1)

    #removing samples with negative values
    train_data_raw_positive = train_data_raw[train_data_raw.min(axis=1)>=0,:]

    #creating numpy arrays to store EPANET simulation output
    train_samples_positive = train_data_raw_positive.shape[0]

    # updating reservoir head in the epanet input
    wn.get_node(1).head_timeseries.base_value = train_data_raw_positive[0,0]

    # updating nodal demand for all nodes in the epanet input
    j=1
    for n in wn.nodes.junction_names:
        wn.get_node(n).demand_timeseries_list[0].base_value = train_data_raw_positive[0,j]
        j=j+1

    # running epanet simulator
    sim = wntr.sim.EpanetSimulator(wn)

    # storing simulation results in 3 matrices
    results = sim.run_sim()

    flowrate = wn.query_link_attribute('flowrate')

    G = wn.get_graph(node_weight=results.node['head'],
                     link_weight=results.link['flowrate'])

    return G, wn, results

if __name__ == "__main__":
    #@ray.remote
    def generate_train_data(cov_mat, nodes_data, ids):
        G, wn, results = graph_generation(cov_mat, nodes_data)

        G = nx.Graph(G)

        nx.write_gpickle({'graph': G},
                         f'../data/training_data_no_leak/network_{ids}')

        print(ids)

    #ray.init(num_cpus=10)

    num_train = 100000
    for ids in range(20500, num_train):
        generate_train_data(covmat_base, base_demands, ids)
        #generate_train_data.remote(covmat_base, base_demands, ids)



    '''
    G, wn, results = graph_generation(covmat_base,base_demands)

    G = nx.Graph(G)

    pos=nx.get_node_attributes(G,'pos')
    node_head=nx.get_node_attributes(G,'weight')
    edge_flowrate = nx.get_edge_attributes(G,'weight')


    node_weigts = [node_head[key][0] for key in node_head.keys()]
    edge_weights = [edge_flowrate[key][0] for key in edge_flowrate.keys()]


    vmin = np.min(node_weigts)
    vmax = np.max(node_weigts)
    cmap = plt.get_cmap('viridis')

    nx.draw(G,pos,with_labels=True, arrows=True, cmap=cmap, width=edge_weights,
            node_color=node_weigts, vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    plt.show()
    '''