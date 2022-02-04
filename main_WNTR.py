import pdb

import numpy as np
import os
import wntr
import networkx as nx
from utils.graph_utils import get_adjacency_matrix
import matplotlib.pyplot as plt
from networkx.convert_matrix import to_numpy_matrix
from networkx.linalg.graphmatrix import incidence_matrix
from wntr.metrics.topographic import algebraic_connectivity
import copy
from scipy.optimize import newton
import pandas as pd

def get_incidence_mat(WNTR_model):
    incidence_mat = np.zeros((34,32))

    for i in range(1,33):
        inlet_ids = WNTR_model.get_links_for_node(str(i), 'INLET')
        outlet_ids = WNTR_model.get_links_for_node(str(i), 'OUTLET')
        for j in inlet_ids:
            incidence_mat[int(j)-1, i-1] = -1
        for j in outlet_ids:
            incidence_mat[int(j)-1, i-1] = 1
    return incidence_mat

def incidence_to_adjacency(incidence_mat):
    adjacency_mat = (np.dot(np.abs(incidence_mat).T, np.abs(incidence_mat)) > 0).astype(int)
    np.fill_diagonal(adjacency_mat, 0)
    return adjacency_mat

def get_edge_lists():
    A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 10, 14, 15, 16, 17, 18, 19, 3,
         20, 21, 20, 23, 24, 25, 26, 27, 23, 28, 29, 30, 31, 32]
    B = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 3, 20,
         21, 22, 23, 24, 25, 26, 27, 16, 28, 29, 30, 31, 32, 25]
    return A, B


# function to create a covariance matrix
def cov_mat_fixed(corr_demands, corr_reservoir_nodes):
    pdb.set_trace()
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

def simulate_no_leak(base_demands):

    #removing samples with negative values
    train_data_raw_positive = base_demands[base_demands.min()>=0,:]

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

    head = results.node['head'].to_numpy().T
    head = np.concatenate((head[-1:, :], head[0:-1, :]), axis=0)
    demand = results.node['demand'].to_numpy().T
    demand = np.concatenate((demand[-1:, :], demand[0:-1, :]), axis=0)
    flow_rate = results.link['flowrate'].to_numpy().T

    G = wn.get_graph(node_weight=results.node['head'],
                     link_weight=results.link['flowrate'],
                     modify_direction=True)
    adjecency_mat = to_numpy_matrix(nx.Graph(G))
    incidence_mat = get_incidence_mat(wn)

    resistance = results.link['friction_factor'].to_numpy().T
    head_loss = results.link['headloss'].to_numpy().T

    return head, demand, flow_rate, resistance, adjecency_mat, incidence_mat


def simulate_with_leak(base_demands, leak, wn):
    # removing samples with negative values
    train_data_raw_positive = base_demands[base_demands.min() >= 0, :]

    # updating reservoir head in the epanet input
    wn.get_node(1).head_timeseries.base_value = train_data_raw_positive[0, 0]

    # updating nodal demand for all nodes in the epanet input
    j = 1
    for n in wn.nodes.junction_names:
        wn.get_node(n).demand_timeseries_list[0].base_value = \
        train_data_raw_positive[0, j]
        j = j + 1

    wn_leak = copy.deepcopy(wn)

    leak_pipe = leak['pipe']
    leak_area = leak['area']

    wn_leak = wntr.morph.link.split_pipe(wn_leak, leak_pipe, 'leak_pipe', 'leak_node')
    leak_node = wn_leak.get_node('leak_node')
    leak_node.add_leak(wn_leak, area=leak_area, start_time=0)
    # running epanet simulator

    sim = wntr.sim.WNTRSimulator(wn_leak)
    # storing simulation results in 3 matrices
    results = sim.run_sim()
    pdb.set_trace()
    print(0.75 * leak_area * np.sqrt(2 * 9.82 * results.node['head']['leak_node'][0]))

    node_pressure_heads = results.node['head']
    pipe_flowrates = copy.deepcopy(results.link['flowrate'])

    pipe_flowrates[f'{leak_pipe}'] = 0.5 * (pipe_flowrates[f'{leak_pipe}']
                                          + pipe_flowrates[f'leak_pipe'])

    G = wn.get_graph(node_weight=node_pressure_heads,
                     link_weight=pipe_flowrates)

    head = results.node['head'].drop('leak_node', axis=1).to_numpy().T
    head = np.concatenate((head[-1:, :], head[0:-1, :]), axis=0)
    demand = results.node['demand'].drop('leak_node', axis=1).to_numpy().T
    demand = np.concatenate((demand[-1:, :], demand[0:-1, :]), axis=0)
    flow_rate = pipe_flowrates.to_numpy()[0:1, 0:-1].T

    G = wn.get_graph(node_weight=results.node['head'],
                     link_weight=results.link['flowrate'],
                     modify_direction=True)
    incidence_mat = get_incidence_mat(wn)
    adjacency_mat = incidence_to_adjacency(incidence_mat)


    demand_df = pd.DataFrame(data=demand.T, columns=range(1, 33))
    head_df = pd.DataFrame(data=head.T, columns=range(1, 33))
    flowrate_df = pd.DataFrame(data=flow_rate.T, columns=range(1, 35))
    G_df = wn.get_graph(node_weight=head_df,
                        link_weight=flowrate_df,
                        modify_direction=True)

    pos = {}
    for i in range(1, 33):
        pos[i] = nx.get_node_attributes(G, 'pos')[str(i)]

    A, B = get_edge_lists()
    df = pd.DataFrame(flow_rate[:, 0:1], columns=['flow_rate'])
    df['a'] = A
    df['b'] = B
    G_lol = nx.from_pandas_edgelist(df, source='a', target='b',
                                    edge_attr='flow_rate')
    nx.set_node_attributes(G_lol, head_df, name='head')
    nx.set_node_attributes(G_lol, demand_df, name='demand')
    nx.set_node_attributes(G_lol, pos, name='pos')


    node_cmap = plt.get_cmap('viridis')
    edge_cmap = plt.get_cmap('viridis')
    nx.draw_networkx_nodes(G=G_lol, pos=pos,
                           node_color=[nx.get_node_attributes(G_lol,name='head')[i][0] for i in range(1,33)],
                           cmap=node_cmap,
                           with_labels=True)
    nx.draw_networkx_edges(G=G_lol, pos=pos,
                           edge_color=[nx.get_edge_attributes(G_lol,name='flow_rate')[key] for key in list(G_lol.edges.keys())],
                           cmap=edge_cmap)
    nx.draw_networkx_labels(G=G_lol, pos=pos)
    plt.show()


    return head, demand, flow_rate, adjacency_mat, incidence_mat




if __name__ == "__main__":

    # Getting path for the input file
    inputfiles_folder_name = 'Input_files_EPANET'
    filename = 'Hanoi_base_demand.inp'
    path_file = os.path.join(inputfiles_folder_name, filename)

    # Reading the input file into EPANET
    inp_file = path_file
    wn = wntr.network.WaterNetworkModel(inp_file)
    wn1 = wntr.network.WaterNetworkModel(inp_file)

    # store no of nodes and links
    num_nodes = len(wn1.node_name_list)
    num_links = len(wn1.link_name_list)

    # create array that contains the base reservoir head and demand at nodes
    base_demands = np.zeros((num_nodes))
    base_demands[0] = wn1.get_node(1).head_timeseries.base_value
    for i in range(1, num_nodes):
        base_demands[i] = wn1.get_node(i + 1).demand_timeseries_list[0].base_value

    # define standard deviation matrix
    std_dev = base_demands * 0.2
    std_dev[0] = base_demands[0] * 0.05

    base_demands[0] = wn1.get_node(1).head_timeseries.base_value

    head, demand, flow_rate, resistance, A, AA = simulate_no_leak(base_demands=base_demands)

    demand_pred = np.dot(-np.transpose(AA), flow_rate)

    x = np.concatenate((flow_rate, head), axis=0)

    G = lambda q: resistance
    LHS_mat = lambda x: np.block([[np.diag(G(x[0:34])[:,0]), -AA],
                              [-np.transpose(AA), np.zeros((32,32))]])
    LHS = lambda x: np.dot(LHS_mat(x), x)
    RHS = np.block([[30*np.ones((34,1))],
                    [np.reshape(base_demands, (32,1))]])

    F = lambda x: LHS(x) - RHS

    '''
    #lol = np.linalg.solve(LHS_mat(flow_rate),RHS)
    #pdb.set_trace()
    rhs_lol = G(flow_rate)*flow_rate
    head_pred = np.linalg.lstsq(AA, rhs_lol)


    sol = newton(F, x0=x)

    plt.figure()
    plt.semilogy(np.abs(F(x)))
    plt.show()
    '''
    leak_pipe = 5
    head_leak, demand_leak, flow_rate, A, AA = simulate_with_leak(base_demands=base_demands,
                                                        leak={'pipe': leak_pipe,
                                                              'area': 0.1},
                                                        wn=wn)

    demand_pred = np.dot(-np.transpose(AA), flow_rate)

    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.plot(range(1,33), demand_leak, label='True Leak Demand')
    #plt.plot(range(1,33), demand, label='True Non-Leak Demand')
    plt.plot(range(1,33), demand_pred, label='Pred Leak Demand')
    plt.axvline(x=leak_pipe, linewidth=2, color='k')
    plt.legend()

    plt.subplot(2,2,2)
    diff = demand-demand_pred
    plt.plot(range(1,33), diff[0:32], label='Demand: No Leak - Leak')
    plt.axvline(x=leak_pipe, linewidth=2, color='k')
    plt.legend()

    print(np.sum(np.abs(diff[0])))

    plt.subplot(2,2,3)
    plt.plot(range(1,33), head, label='No Leak Head')
    plt.plot(range(1,33), head_leak, label='Leak Head')
    plt.axvline(x=leak_pipe, linewidth=2, color='k')
    plt.legend()

    plt.subplot(2,2,4)
    diff = head_leak-head
    plt.plot(range(1,33), diff[0:32], label='Head: No Leak - Leak')
    plt.axvline(x=leak_pipe, linewidth=2, color='k')
    plt.legend()
    plt.show()

