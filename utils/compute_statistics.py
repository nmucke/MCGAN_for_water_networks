import pdb
import torch

class GANStatistics():
    def __init__(self):
        super(GANStatistics, self).__init__()

def get_statistics_from_latent_samples(z_samples,
                                       generator,
                                       transform=None,
                                       gan_with_leak=True):
    gen_samples = generator(z_samples)

    if transform is not None:
        gen_samples = transform(gen_samples)

    if gan_with_leak:
        gen_samples, gen_leak_demand, gen_leak_pipe = \
            gen_samples[:, 0:66], gen_samples[:, 66:67], gen_samples[:, -34:]
    else:
        gen_samples = gen_samples[:, 0:66]

    gen_flow_rate_samples = gen_samples[:, 0:34]
    gen_flow_rate_mean = torch.mean(gen_flow_rate_samples, dim=0)
    gen_flow_rate_std = torch.std(gen_flow_rate_samples, dim=0)/torch.abs(gen_flow_rate_mean)

    gen_head_samples = gen_samples[:, 34:66]
    gen_head_mean = torch.mean(gen_head_samples, dim=0)
    gen_node_std = torch.std(gen_head_samples, dim=0)/torch.abs(gen_head_mean)

    flow_rate_dict = {"mean": gen_flow_rate_mean,
                      "std": gen_flow_rate_std,
                      "samples": gen_flow_rate_samples}
    head_dict = {"mean": gen_head_mean,
                 "std": gen_node_std,
                 "samples": gen_head_samples}

    output_dict = {'flow_rate': flow_rate_dict,
                   'head': head_dict}
    if gan_with_leak:
        demand_mean = torch.mean(gen_leak_demand,dim=0)
        demand_std = torch.std(gen_leak_demand,dim=0)

        leak_pipe_estimate = torch.mean(gen_leak_pipe,dim=0)
        leak_pipe_estimate = torch.argmax(leak_pipe_estimate) + 2
        leak ={'samples': gen_leak_pipe,
               'estimate': leak_pipe_estimate.item(),
               'demand_mean': demand_mean,
               'demand_std': demand_std,
               'demand_samples': gen_leak_demand}

        output_dict['leak'] = leak

    return output_dict

def get_demand_statistics(data, incidence_mat):

    demand_samples = torch.matmul(-incidence_mat.T,
                                  data['flow_rate']['samples'].T)
    demand_samples = demand_samples.T
    demand_mean = torch.mean(demand_samples, dim=0)
    demand_std = torch.std(demand_samples, dim=0)/torch.abs(demand_mean)

    return {'samples': demand_samples,
            'mean': demand_mean,
            'std': demand_std}