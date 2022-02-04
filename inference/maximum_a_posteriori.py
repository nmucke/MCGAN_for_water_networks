import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb

def compute_MAP(z, observations, generator, obs_operator, num_iter=3000):
    optimizer = optim.Adam([z], lr=1e-2)

    loss = nn.MSELoss()
    for i in range(num_iter):
        optimizer.zero_grad()
        error = loss(observations, obs_operator(generator(z)))
        error.backward()
        optimizer.step()

    return z.detach()
