import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch


class MLP(nn.Module):
    """Dense Neural Network"""
    def __init__(self, input_dim, hidden_dims, output_dim,
                 activation=nn.LeakyReLU(), dropout_rate=0.5):
        """Initialize dense neural net
        args:
            input_dim (int): Input dimension.
            hidden_dims (list of ints): List of dimension of each hidden layer.
            output_dim (int): Output dimension
            activation (torch.nn.activation): Activation function.
            dropout_rate (float): Dropout rate.
        """
        super(MLP, self).__init__()

        self.num_hidden_layers = len(hidden_dims)
        self.activation = activation

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.linear_input = nn.Linear(in_features=input_dim,
                                      out_features=hidden_dims[0],
                                      bias=True)

        self.linear = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        for i in range(1, self.num_hidden_layers):
            self.batchnorm.append(nn.BatchNorm1d(num_features=hidden_dims[i - 1]))
            self.linear.append(nn.Linear(in_features=hidden_dims[i - 1],
                                         out_features=hidden_dims[i],
                                         bias=True))

        self.linear_output = nn.Linear(in_features=hidden_dims[-1],
                                       out_features=output_dim,
                                       bias=False)

    def forward(self, x):
        """Forward propagation"""

        x = self.linear_input(x)
        x = self.dropout(x)
        x = self.activation(x)

        for (linear, batchnorm) in zip(self.linear, self.batchnorm):
            x = batchnorm(x)
            x = linear(x)
            x = self.dropout(x)
            x = self.activation(x)

        return self.linear_output(x)

class MessagePassing(nn.Module):
    """Neural Network class."""

    def __init__(self, adjacency_matrix):
        super(MessagePassing, self).__init__()

        self.A = adjacency_matrix


    def forward(self, x):
        """Forward pass."""

        return x

class Aggregate(nn.Module):
    """Neural Network class."""

    def __init__(self, aggregate_type='average'):
        super(Aggregate, self).__init__()

        if aggregate_type == 'average':
                self.aggregate_function = nn.Mean(dim=1)
        elif aggregate_type == 'sum':
            self.aggregate_function = nn.Sum(dim=1)


    def forward(self, x):
        """Forward pass."""
        return self.aggregate_function(x)

class Update(nn.Module):
    """Neural Network class."""

    def __init__(self, ):
        super(Update, self).__init__()


    def forward(self, x):
        """Forward pass."""

        return x