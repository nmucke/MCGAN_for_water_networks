import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch


class Generator(nn.Module):
    """Neural Network class."""

    def __init__(self, latent_dim, par_dim, output_dim,
                 n_neurons, activation, leak):
        """Initialize as subclass of nn.Module, inherit its methods."""
        super(Generator, self).__init__()

        self.input_dim = latent_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons
        self.n_layers = len(n_neurons)
        self.activation = activation
        self.par_dim = par_dim
        self.tanh = nn.Tanh()

        self.in_layer = nn.Linear(in_features=self.input_dim,
                                  out_features=self.n_neurons[0])

        self.hidden_layers = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        for i in range(len(self.n_neurons)-1):
            self.hidden_layers.append(nn.Linear(in_features=self.n_neurons[i],
                                                out_features=self.n_neurons[i+1]))
            self.batchnorm.append(nn.BatchNorm1d(self.n_neurons[i+1]))

        self.out_layer_state = nn.Linear(in_features=self.n_neurons[-1],
                                   out_features=self.output_dim)

        if leak:
            self.leak = leak
            self.softmax = nn.Softmax(dim=1)
            self.out_layer_par = nn.Linear(in_features=self.n_neurons[-1],
                                           out_features=self.par_dim)

    def forward(self, x):
        """Forward pass."""

        x = self.in_layer(x)
        x = self.activation(x)

        for _, (hidden_layer, batchnorm) \
                in enumerate(zip(self.hidden_layers, self.batchnorm)):

            x = hidden_layer(x)
            x = self.activation(x)
            x = batchnorm(x)

        state = self.tanh(self.out_layer_state(x))

        if self.leak:
            par = self.softmax(self.out_layer_par(x))
            return torch.cat([state,par], dim=1)
        else:
            return state

class Critic(nn.Module):
    """Neural Network class."""

    def __init__(self, input_dim, n_neurons, activation):
        """Initialize as subclass of nn.Module, inherit its methods."""
        super(Critic, self).__init__()

        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.n_layers = len(n_neurons)
        self.activation = activation

        self.in_layer = nn.Linear(in_features=self.input_dim,
                                  out_features=self.n_neurons[0])

        self.hidden_layers = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        for i in range(len(self.n_neurons)-1):
            self.hidden_layers.append(nn.Linear(in_features=self.n_neurons[i],
                                                out_features=self.n_neurons[i+1]))
            self.batchnorm.append(nn.BatchNorm1d(self.n_neurons[i+1]))

        self.out_layer = nn.Linear(in_features=self.n_neurons[-1],
                                   out_features=1,
                                   bias=False)

    def forward(self, x):
        """Forward pass."""

        x = self.in_layer(x)
        x = self.activation(x)

        for _, (hidden_layer, batchnorm) in \
                enumerate(zip(self.hidden_layers, self.batchnorm)):

            x = hidden_layer(x)
            x = self.activation(x)

        return self.out_layer(x)