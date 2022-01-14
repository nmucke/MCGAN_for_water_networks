import pdb
import torch

class GANStatistics():
    def __init__(self):
        super(GANStatistics, self).__init__()

def get_statistics_from_latent_samples(z_samples,
                                       generator,
                                       separate_features=False,
                                       transform=None,
                                       gan_with_leak=True):
    gen_samples = generator(z_samples)

    if gan_with_leak:
        gen_samples, gen_leak_pipe = gen_samples[:, 0:66], gen_samples[:, -33:]
    else:
        gen_samples = gen_samples[:, 0:66]

    if transform is not None:
        gen_samples = transform(gen_samples)

    if not separate_features:
        gen_mean = torch.mean(gen_samples, dim=0)
        gen_std = torch.std(gen_samples, dim=0)/torch.abs(gen_mean)

        output_dict = {"gen_mean": gen_mean,
                       "gen_std": gen_std,
                       "gen_samples":gen_samples}

        if gan_with_leak:
            output_dict["gen_leak_pipe"] = gen_leak_pipe

            leak_pipe_estimate = torch.mean(gen_leak_pipe,dim=0)
            leak_pipe_estimate = torch.argmax(leak_pipe_estimate) + 2
            output_dict["gen_leak_pipe_estimate"] = leak_pipe_estimate.item()

    else:
        gen_node_samples = gen_samples[:, 0:32]
        gen_node_mean = torch.mean(gen_node_samples, dim=0)
        gen_node_std = torch.std(gen_node_samples, dim=0)/torch.abs(gen_node_mean)

        gen_edge_samples = gen_samples[:, -34:]
        gen_edge_mean = torch.mean(gen_edge_samples, dim=0)
        gen_edge_std = torch.std(gen_edge_samples, dim=0)/torch.abs(gen_edge_mean)


        output_dict = {"gen_node_mean": gen_node_mean,
                        "gen_edge_mean": gen_edge_mean,
                        "gen_node_std": gen_node_std,
                        "gen_edge_std": gen_edge_std,
                        "gen_node_samples": gen_node_samples,
                        "gen_edge_samples": gen_edge_samples}

        if gan_with_leak:
            output_dict["gen_leak_pipe"] = gen_leak_pipe

            leak_pipe_estimate = torch.mean(gen_leak_pipe,dim=0)
            leak_pipe_estimate = torch.argmax(leak_pipe_estimate) + 2
            output_dict["gen_leak_pipe_estimate"] = leak_pipe_estimate.item()

    return output_dict