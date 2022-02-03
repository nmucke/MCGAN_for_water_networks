import pdb
import os
import wntr
import copy
import ray
from wntr.network.model import *
from wntr.sim.solvers import *
import matplotlib.pyplot as plt

# function to create a covariance matrix
def cov_mat_fixed(corr_demands, corr_reservoir_nodes):
    N = num_nodes

    mat_corr = np.zeros((N, N))  # initializing matrix

    mat = np.full((N - 1, N - 1),
                  corr_demands)  # step 1 for a symmetric matrix of n-1 by n-1
    mat_symm = (mat + mat.T) / 2  # step 2

    diag = np.ones(N)  # setting up the diagonal matrix, variance of nodal demands
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


def graph_generation(cov_mat,nodes_data, leak, wn):

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

    wn_leak = copy.deepcopy(wn)

    leak_pipe = leak['pipe']
    leak_area = leak['area']

    wn_leak = wntr.morph.link.split_pipe(wn_leak, leak_pipe, 'leak_pipe', 'leak_node')
    leak_node = wn_leak.get_node('leak_node')
    leak_node.add_leak(wn_leak, area=leak_area, start_time=0)
    # running epanet simulator

    #lol = HydraulicModel(wn)
    sim = wntr.sim.WNTRSimulator(wn_leak)
    # storing simulation results in 3 matrices
    results = sim.run_sim()

    G_leak = wn_leak.get_graph(node_weight=results.node['head'],
                     link_weight=results.link['flowrate'])

    node_pressure_heads = results.node['head']
    pipe_flowrates = copy.deepcopy(results.link['flowrate'])

    pipe_flowrates[f'{leak_pipe}'] = 0.5 * (pipe_flowrates[f'{leak_pipe}']
                                          + pipe_flowrates[f'leak_pipe'])

    G = wn.get_graph(node_weight=node_pressure_heads,
                     link_weight=pipe_flowrates)

    return G, wn, results



if __name__ == "__main__":
    # Getting path for the input file
    inputfiles_folder_name = '../Input_files_EPANET'
    filename = 'Hanoi_base_demand.inp'
    path_file = os.path.join(inputfiles_folder_name,filename)

    # Load the wntr model
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

    # Covmat for all experiments
    covmat_base = cov_mat_fixed(0.6, 0.0)

    # Column names for the dataframe that will store results. Note that the
    # Reservoir Demand and Head comes after all
    # other nodes. Naming is done accordingly.

    col_names = ['Demand' + str(i) for i in range(2, 33)] + \
                ['Demand_Reservoir'] + \
                ['Node_head' + str(i) for i in range(2, 33)] + \
                ['Res_Head'] + \
                ['Link_flow' + str(i) for i in range(1, 35)]

    #G, wn, results = graph_generation(covmat_base, base_demands, leak=1, wn=wn)

    #@ray.remote
    def generate_train_data(cov_mat, nodes_data, leak, wn, ids):
        G, wn, results = graph_generation(cov_mat, nodes_data, leak, wn)

        #G = nx.Graph(G)
        save_dict = {'graph': G,
                     'leak_pipe': leak['pipe'],
                     'leak_area': leak['area']}
        print(ids)

        nx.write_gpickle(save_dict, f'../data/training_data_with_leak/network_{ids}')

    #ray.init(num_cpus=30)
    num_train = 200000
    leak_pipes = np.random.randint(low=2, high=35, size=num_train)
    leak_areas = np.random.uniform(low=0.01, high=0.1, size=num_train)
    for ids in range(196000, num_train):
        leak = {'pipe': leak_pipes[ids],
                'area': leak_areas[ids]}
        generate_train_data(covmat_base, base_demands, leak, wn, ids)
        #generate_train_data.remote(covmat_base, base_demands, leak, wn, ids)
