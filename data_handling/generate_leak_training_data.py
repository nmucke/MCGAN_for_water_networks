import pdb
import os
import wntr
import copy
import ray
from wntr.network.model import *
from wntr.sim.solvers import *

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
    wn_leak_out = copy.deepcopy(wn)

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

    G = wn.get_graph(node_weight=results.node['head'],
                     link_weight=results.link['flowrate'])
    '''
    lol = list(G.edges)
    for i in lol:
        print(i[2])

    lal = list(G.nodes)
    for i in lal:
        print(i)
    pdb.set_trace()
    '''
    #G = nx.Graph(G)
    '''
    pos = nx.get_node_attributes(G, 'pos')
    node_head = nx.get_node_attributes(G, 'weight')
    edge_flowrate = nx.get_edge_attributes(G, 'weight')

    node_weigts = [node_head[key][0] for key in node_head.keys()]
    edge_weights = [edge_flowrate[key][0] for key in edge_flowrate.keys()]

    head = np.asarray(node_weigts).reshape((32,1))
    A = incidence_matrix(G).todense()
    A = np.asarray(A)
    _Hw_k = 10.666829500036352

    R = np.zeros((34,34))
    for i in range(2,35):
        R[i-2,i-2] =_Hw_k * (wn.get_link(str(i)).roughness**(-1.852)) * (wn.get_link(str(i)).diameter ** (-4.871)) * wn.get_link(str(i)).length
    R[-1,-1] = _Hw_k * (wn.get_link(str(1)).roughness**(-1.852)) * (wn.get_link(str(1)).diameter ** (-4.871)) * wn.get_link(str(1)).length
    E = np.zeros((32,1))

    for i in range(2,33):
        E[i-2,0] = wn.get_node(str(i)).elevation
    #E[-1,0] = wn.get_node(str(1)).elevation


    RHS = np.dot(np.transpose(A),head) + np.dot(np.transpose(A),E)
    LHS = lambda x: np.dot(R,np.power(np.abs(x),1.852-1))
    x0 = np.asarray(edge_weights)#np.zeros((34, ))
    fun = lambda x: LHS(x)-RHS[:,0]
    pdb.set_trace()
    sol = fsolve(fun, x0=x0)
    pdb.set_trace()
    sol = np.linalg.solve(R,RHS)
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
    plt.show()
    pdb.set_trace()
    '''
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

        nx.write_gpickle(save_dict, f'../data/training_data_with_leak_small/network_{ids}')

    ray.init(num_cpus=30)
    num_train = 100000
    leak_pipes = np.random.randint(low=2, high=35, size=num_train)
    leak_areas = np.random.uniform(low=0.001, high=0.01, size=num_train)
    for ids in range(24000, num_train):
        leak = {'pipe': leak_pipes[ids],
                'area': leak_areas[ids]}
        generate_train_data(covmat_base, base_demands, leak, wn, ids)
        #generate_train_data.remote(covmat_base, base_demands, leak, wn, ids)


    '''
    G = nx.Graph(G)

    pos = nx.get_node_attributes(G, 'pos')
    node_head = nx.get_node_attributes(G, 'weight')
    edge_flowrate = nx.get_edge_attributes(G, 'weight')

    node_weigts = [node_head[key][0] for key in node_head.keys()]
    edge_weights = [edge_flowrate[key][0] for key in edge_flowrate.keys()]

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
    '''

