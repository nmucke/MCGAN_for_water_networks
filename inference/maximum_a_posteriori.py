import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb

def compute_MAP(z, observations, generator, obs_operator):
    optimizer = optim.Adam([z], lr=1e-2, weight_decay=1.)

    loss = nn.MSELoss()
    for i in range(1000):
        optimizer.zero_grad()
        error = loss(observations, obs_operator(generator(z)[0, 0:66]))
        error.backward()
        optimizer.step()

    return z.detach()
