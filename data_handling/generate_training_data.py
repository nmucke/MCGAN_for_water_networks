import pdb

import numpy as np
import os
import wntr
import networkx as nx
import copy
import pandas as pd
import matplotlib.pyplot as plt
from utils.graph_utils import get_incidence_mat, incidence_to_adjacency, get_edge_lists
from scipy.optimize import newton, curve_fit


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

def fourier_series(t, a, b):
    out = np.zeros(t.shape)
    for i in range(1, len(a)):
        out += a[i] * np.cos(2*np.pi*i/24 * t)
        out += b[i-1] * np.sin(2*np.pi*i/24 * t)
    return out + a[0]

def generate_demand(start_time, end_time, time_steps):

    time = np.arange(1, 25)

    alpha1 = [0.030, 0.007, 0.004, 0.006, 0.005, 0.002, 0.005, 0.037, 0.065, 0.067, 0.067, 0.059,
             0.068, 0.079, 0.081, 0.054, 0.027, 0.027, 0.030, 0.045, 0.077, 0.066, 0.050, 0.041]
    alpha2 = [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000 ,0.081, 0.098 ,0.086, 0.116,
              0.167 ,0.144 ,0.080 ,0.066, 0.106 ,0.051, 0.002 ,0.002 ,0.000 ,0.000, 0.000 ,0.000]
    alpha3 = [0.006, 0.001, 0.003, 0.001, 0.003, 0.002 ,0.007, 0.063, 0.101, 0.063, 0.076, 0.066,
              0.092, 0.075, 0.074, 0.045, 0.039, 0.052, 0.047 ,0.043, 0.064, 0.046, 0.017, 0.015]
    alpha4 = [0.016, 0.007, 0.006, 0.003, 0.004, 0.007, 0.013, 0.057, 0.076, 0.054, 0.054, 0.071,
              0.060, 0.076, 0.078, 0.062 ,0.048, 0.052, 0.041, 0.052, 0.068, 0.042, 0.033, 0.022]

    alpha1 = np.asarray(alpha1)
    alpha2 = np.asarray(alpha2)
    alpha3 = np.asarray(alpha3)
    alpha4 = np.asarray(alpha4)

    alpha = np.stack((alpha1, alpha2, alpha3, alpha4),axis=1)
    alpha_mean = np.mean(alpha, axis=1)
    #sol = curve_fit(fourier_series, time, alpha_mean, coef)

    a = [np.mean(alpha_mean)]
    b = []
    for i in range(1, 8):
        aa = 2*np.sum(alpha_mean*np.cos(2*np.pi*i/24 * time))
        bb = 2*np.sum(alpha_mean*np.sin(2*np.pi*i/24 * time))

        a.append(aa)
        b.append(bb)

    a = np.asarray(a)
    b = np.asarray(b)


    time_new = np.linspace(start_time, end_time, time_steps)
    fourier = fourier_series(time_new, a, b)

    time_new_noise = time_new + np.random.normal(0, 0.25, time_new.shape)
    a_noise = a + np.random.normal(0, 0.075, a.shape)
    b_noise = b + np.random.normal(0, 0.075, b.shape)
    fourier_noise = fourier_series(time_new_noise, a_noise, b_noise)
    #fourier_noise += np.abs(np.min(fourier_noise)) + 5e-2
    #fourier_noise /= np.max(fourier_noise)
    fourier_noise *= 0.05
    #fourier_noise += np.random.normal(0, 0.05, fourier_noise.shape)
    #fourier_noise = np.abs(fourier_noise)

    return fourier_noise

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
        leak_node.add_leak(wn_leak, area=leak_area, start_time=10*3600)
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

    train_data = False
    with_leak = True
    num_samples = 1000

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

    sample_ids = range(0,num_samples)
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
