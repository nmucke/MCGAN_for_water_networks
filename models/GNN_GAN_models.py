import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch


class MLP(nn.Module):
    """Neural Network class."""

    def __init__(self, ):
        super(MLP, self).__init__()


    def forward(self, x):
        """Forward pass."""

        return x

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

    def __init__(self, ):
        super(Aggregate, self).__init__()


    def forward(self, x):
        """Forward pass."""

        return x

class Update(nn.Module):
    """Neural Network class."""

    def __init__(self, ):
        super(Update, self).__init__()


    def forward(self, x):
        """Forward pass."""

        return x