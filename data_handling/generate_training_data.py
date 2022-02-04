import pdb

import numpy as np
import os
import wntr
import networkx as nx
import copy
import pandas as pd
import matplotlib.pyplot as plt
from utils.graph_utils import get_incidence_mat, incidence_to_adjacency, get_edge_lists

def cov_mat_fixed(corr_demands, corr_reservoir_nodes, num_nodes=32):
    N = num_nodes

    mat_corr = np.zeros((N, N))

    mat = np.full((N - 1, N - 1),
                  corr_demands)
    mat_symm = (mat + mat.T) / 2

    diag = np.ones(N)
    np.fill_diagonal(mat_symm, diag)
    mat_corr[1:, 1:] = mat_symm

    mat_corr[0, 0] = 1

    top = np.full((N - 1),corr_reservoir_nodes)

    mat_corr[0, 1:] = top
    mat_corr[1:, 0] = top

    Diag = np.diag(std_dev)
    cov_mat = Diag * mat_corr * Diag

    return cov_mat

def simulate_WDN(demands, leak, wn):

    # removing samples with negative values
    demand_data = demands[demands.min() >= 0]

    # updating reservoir head in the epanet input
    wn.get_node(1).head_timeseries.base_value = demand_data[0, 0]

    # updating nodal demand for all nodes in the epanet input
    j = 1
    for n in wn.nodes.junction_names:
        wn.get_node(n).demand_timeseries_list[0].base_value = demand_data[0, j]
        j = j + 1

    if leak is not None:
        wn_leak = copy.deepcopy(wn)

        leak_pipe = leak['pipe']
        leak_area = leak['area']

        wn_leak = wntr.morph.link.split_pipe(wn_leak, leak_pipe, 'leak_pipe', 'leak_node')
        leak_node = wn_leak.get_node('leak_node')
        leak_node.add_leak(wn_leak, area=leak_area, start_time=0)
        # running epanet simulator

        sim = wntr.sim.WNTRSimulator(wn_leak)
    else:
        sim = wntr.sim.EpanetSimulator(wn)

    results = sim.run_sim()

    G = wn.get_graph()
    pipe_flowrates = copy.deepcopy(results.link['flowrate'])

    if leak is not None:
        leak['demand'] = results.node['leak_demand']['leak_node'][0]

        pipe_flowrates[f'{leak_pipe}'] = 0.5 * (pipe_flowrates[f'{leak_pipe}']
                                                + pipe_flowrates[f'leak_pipe'])
        head = results.node['head'].drop('leak_node', axis=1).to_numpy().T
        demand = results.node['demand'].drop('leak_node', axis=1).to_numpy().T
    else:
        head = results.node['head'].to_numpy().T
        demand = results.node['demand'].to_numpy().T

    head = np.concatenate((head[-1:, :], head[0:-1, :]), axis=0)
    head_df = pd.DataFrame(data=head.T, columns=range(1, 33))
    demand = np.concatenate((demand[-1:, :], demand[0:-1, :]), axis=0)
    demand_df = pd.DataFrame(data=demand.T, columns=range(1, 33))


    if leak is not None:
        flow_rate = pipe_flowrates.to_numpy()[0:1, 0:-1].T
    else:
        flow_rate = pipe_flowrates.to_numpy()[0:1, :].T
    flowrate_df = pd.DataFrame(data=flow_rate.T, columns=range(1, 35))

    incidence_mat = get_incidence_mat(wn)
    adjacency_mat = incidence_to_adjacency(incidence_mat)

    pos = {}
    for i in range(1, 33):
        pos[i] = nx.get_node_attributes(G, 'pos')[str(i)]

    A, B = get_edge_lists()
    df = pd.DataFrame(flow_rate[:, 0:1], columns=['flow_rate'])
    df['a'] = A
    df['b'] = B
    G = nx.from_pandas_edgelist(df, source='a', target='b',
                                    edge_attr='flow_rate')
    nx.set_node_attributes(G, head_df, name='head')
    nx.set_node_attributes(G, demand_df, name='demand')
    nx.set_node_attributes(G, pos, name='pos')

    if leak is not None:
        result_dict = {'WNTR_results': results,
                       'graph': G,
                       'head': head_df,
                       'demand': demand_df,
                       'flow_rate': flowrate_df,
                       'leak': leak}
    else:
        result_dict = {'WNTR_results': results,
                       'graph': G,
                       'head': head_df,
                       'demand': demand_df,
                       'flow_rate': flowrate_df}
    return result_dict

if __name__ == "__main__":

    train_data = True
    with_leak = True
    num_samples = 200000

    if train_data:
        if with_leak:
            data_save_path = f'../data/training_data_with_leak/network_'
        else:
            data_save_path = f'../data/training_data_no_leak/network_'
    else:
        if with_leak:
            data_save_path = f'../data/test_data_with_leak/network_'
        else:
            data_save_path = f'../data/test_data_no_leak/network_'

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
    cov_mat = cov_mat_fixed(0.6,0.0)

    sample_ids = range(140000,num_samples)
    if with_leak:
        leak_pipes = np.random.randint(low=1, high=35, size=num_samples)
        leak_areas = np.random.uniform(low=0.01, high=0.1, size=num_samples)
        for id, leak_pipe, leak_area in zip(sample_ids, leak_pipes, leak_areas):
            demands = np.random.multivariate_normal(base_demands,cov_mat,1)
            result_dict_leak = simulate_WDN(demands=demands[0],
                                            leak={'pipe': leak_pipe,
                                                  'area': leak_area},
                                            wn=wn)
            nx.write_gpickle(result_dict_leak, f'{data_save_path}{id}')

            if id % 1000 == 0:
                print(id)

    else:
        for id in sample_ids:
            demands = np.random.multivariate_normal(base_demands,cov_mat,1)
            result_dict = simulate_WDN(demands=demands[0],
                                       leak=None,
                                       wn=wn)
            nx.write_gpickle(result_dict, f'{data_save_path}{id}')

            if id % 1000 == 0:
                print(id)
