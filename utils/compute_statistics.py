import pdb
import torch

class GANStatistics():
    def __init__(self):
        super(GANStatistics, self).__init__()

def get_statistics_from_latent_samples(z_samples,
                                       generator,
                                       separate_features=False,
                                       transform=None):
    gen_samples = generator(z_samples)
    gen_samples, gen_leak_pipe = gen_samples[:, 0:66], gen_samples[:, -33:]

    if transform is not None:
        gen_samples = transform(gen_samples)

    if not separate_features:
        gen_mean = torch.mean(gen_samples, dim=0)
        gen_std = torch.std(gen_samples, dim=0)

        return {"gen_leak_pipe": gen_leak_pipe,
                "gen_mean": gen_mean,
                "gen_std": gen_std,
                "gen_samples":gen_samples}

    else:
        gen_node_samples = gen_samples[:, 0:32]
        gen_node_mean = torch.mean(gen_node_samples, dim=0)
        gen_node_std = torch.std(gen_node_samples, dim=0)

        gen_edge_samples = gen_samples[:, -34:]
        gen_edge_mean = torch.mean(gen_edge_samples, dim=0)
        gen_edge_std = torch.std(gen_edge_samples, dim=0)

        return {"gen_leak_pipe": gen_leak_pipe,
                "gen_node_mean": gen_node_mean,
                "gen_edge_mean": gen_edge_mean,
                "gen_node_std": gen_node_std,
                "gen_edge_std": gen_edge_std,
                "gen_node_samples": gen_node_samples,
                "gen_edge_samples": gen_edge_samples}