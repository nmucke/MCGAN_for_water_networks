
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
from itertools import combinations, combinations_with_replacement, product



# Getting path for the input file
inputfiles_folder_name = 'Input_files_EPANET'
filename = 'Hanoi_leak_14.inp'
path_file = os.path.join(inputfiles_folder_name,filename)


# Getting path for the 'No leak datafile' that contains expected states of given nodal demands and resulting flows and pressures
# across the WDN
data_folder_name = 'Data_files'
data_filename = 'data_base_demand_leakgen.csv'
path_data_file = data_folder_name + '/' + data_filename

# Reading the input file into EPANET
inp_file = path_file
wn = wntr.network.WaterNetworkModel(inp_file)
wn1 = wntr.network.WaterNetworkModel(inp_file)

sim_leak = wntr.sim.EpanetSimulator(wn1)
results = sim_leak.run_sim()
df_demand = results.node['demand']
df_head = results.node['head']
df_flow = results.link['flowrate']

# Use the column names from the results in cell above and create empty dataframes to store results of
# leak experiment
demand_data = pd.DataFrame(columns = df_demand.columns)
leak_demand_data = pd.DataFrame(columns = df_demand.columns)
head_data = pd.DataFrame(columns = df_demand.columns)
flow_data = pd.DataFrame(columns = df_flow.columns)

# 'Expected states' stand for 'No Leak' demand cases that are being used for leak simulation.
# The underlying demand, resulting flow and pressure without any leak are stored along with the
# resulting flow and pressure with leak.

expected_states = pd.read_csv(path_data_file)
expected_states = expected_states.sample(frac=0.25)
expected_states = expected_states.reset_index(drop=True)
expected_state_data = pd.DataFrame(columns=expected_states.columns)
expected_states_np = np.array(expected_states)

# initializing some key values
num_nodes = 32
num_links = 34
train_samples = 10#expected_states_np.shape[0] # No of demand scenarios to be fed to EPANET with leak in place
num_total_leaks = 1 # For current set of experiments

# create list of names of all nodes and links
link_name = wn1.link_name_list
node_name = wn1.node_name_list
leak_node_name = node_name[num_nodes-1:-1] # The leak node is assigned number one greater than the last normal node
leaking_node = pd.DataFrame(columns = leak_node_name)

#Creating more empty dataframes to store results: one for storing the demand scenario being fed into EPANET during
# the experiment, another to store the area of the leak(s) that are active during the experiment
demand_data_in = pd.DataFrame(columns = df_demand.columns[:-1])
area_in = pd.DataFrame(columns=leak_node_name)
pdb.set_trace()

# below code creates a list of nCp combinations for 'n' links with leak nodes
# and 'p' total leaks in the system.
# Here we have assumed that EACH PIPE CAN HAVE ONLY ONE LEAK AT A TIME. Since
# we have only one leak,
# its a single element list.
def leak_combs(num_tot_leaks):
    combs = list(combinations(leak_node_name, num_tot_leaks))

    return combs


# For a given combination of leak nodes, say 2 leak nodes, each node may have a range of leak areas. The code
# below first creates a list of areas and then creates nPc combinations where 'n' is the no of areas and 'c'
# is no of leaks
def leak_area_combs(area_list):
    combs = list(combinations_with_replacement(area_list,num_total_leaks))
    return combs

# A hole of radius 2 cm would have an area of around 1e-3 m2 and hole with radius 12 cm would have
# around 50e-3 m2. We are assuming a leak in this area range. Default emmitter coefficient is 0.75

area_list = leak_area_combs([1e-1])

lol = 0
k = 0
for i in range(train_samples):
    wn = wntr.network.WaterNetworkModel(inp_file)
    # set up the reservoir head as per 'No leak datafile'
    wn.get_node(1).head_timeseries.base_value = expected_states_np[i, 63]

    # set up demands across junction nodes as per 'No leak datafile'
    for n in range(2, num_nodes + 1):
        wn.get_node(n).demand_timeseries_list[0].base_value = \
        expected_states_np[i, n - 2]

        # Loop 1: over each of the leakage node combinations in the network
    for leak_nodes in leak_combs(num_total_leaks):

        # Loop 3: over each of the Emitter coefficient value combination (its
        # a fixed list given no of leaks)
        for a in range(len(area_list)):

            wn_sim = copy.deepcopy(wn)
            leaking_node_data = []  # to store leak node names and
            # corresponding ECs

            # Loop 4: over each of the leakage node for a leakage node
            # combination and EC combination
            for b in range(num_total_leaks):
                wn_sim.get_node(leak_nodes[b]).add_leak(wn_sim,
                                                        area=area_list[a][b],
                                                        start_time=0)
                leaking_node_data.append(leak_nodes[b])
                leaking_node_data.append(area_list[a][b])

            l_leak = []
            for leaknode in leak_node_name:
                l_leak.append(wn_sim.get_node(leaknode).leak_area)
            area_in.loc[k] = l_leak

            sim = wntr.sim.WNTRSimulator(wn_sim)
            results = sim.run_sim()
            demand_data.loc[k] = results.node['demand'].loc[0]
            leak_demand_data.loc[k] = results.node['leak_demand'].loc[0]
            head_data.loc[k] = results.node['head'].loc[0]
            flow_data.loc[k] = results.link['flowrate'].loc[0]
            expected_state_data.loc[k] = expected_states.loc[i]

            G = wn.get_graph(node_weight=results.node['head'],
                             link_weight=results.link['flowrate'])

            G = nx.Graph(G)

            pos = nx.get_node_attributes(G, 'pos')
            node_head = nx.get_node_attributes(G, 'weight')
            edge_flowrate = nx.get_edge_attributes(G, 'weight')

            node_weigts = [node_head[key][0] for key in node_head.keys()]
            edge_weights = [edge_flowrate[key][0] for key in
                            edge_flowrate.keys()]

            vmin = np.min(node_weigts)
            vmax = np.max(node_weigts)
            cmap = plt.get_cmap('viridis')

            nx.draw(G, pos, with_labels=True, arrows=True, cmap=cmap,
                    width=edge_weights,
                    node_color=node_weigts, vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                       norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm)
            #plt.show()

            G = nx.Graph(G)

            nx.write_gpickle(G, f'leak_data/base_demand_with_leak{i}')

            k = k + 1

leak_demand = demand_data.copy()
leak_demand_leak = leak_demand_data.copy()
leak_flow = flow_data.copy()
leak_head = head_data.copy()
leak_area = area_in.copy()
leak_expected = expected_state_data.copy()

leakheadnames = []
for x in list(leak_head):
    y = 'leak_head_' + x
    leakheadnames.append(y)

leak_head.columns = leakheadnames

leakflownames = []
for x in list(leak_flow.columns):
    y = 'leak_flow_' + x
    leakflownames.append(y)

leak_flow.columns = leakflownames

leak_area.columns = ['leak_area']

leakage_demand = pd.DataFrame(columns=['leakage_demand'])
leakage_demand['leakage_demand'] = leak_demand_leak['33']

leak_combined = pd.concat(
        (leak_area, leakage_demand, leak_head, leak_flow, leak_expected),
        axis=1)

'''

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

        nx.write_gpickle(G, f'training_data/base_demand_{ids}')

    #ray.init(num_cpus=20)

    num_train = 100000
    for ids in range(num_train):
        generate_train_data(covmat_base, base_demands, ids)




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