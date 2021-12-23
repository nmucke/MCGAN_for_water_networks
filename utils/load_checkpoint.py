import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch

def load_checkpoint(checkpoint_path, generator=None, critic=None,
                    generator_optimizer=None, critic_optimizer=None):
    checkpoint = torch.load(checkpoint_path)

    if generator_optimizer is not None:
        generator_optimizer.load_state_dict(
                checkpoint['generator_optimizer_state_dict'])

    if critic_optimizer is not None:
        critic_optimizer.load_state_dict(
                checkpoint['critic_optimizer_state_dict'])
    if generator is not None:
        generator.load_state_dict(checkpoint['generator_state_dict'])

    if critic is not None:
        critic.load_state_dict(checkpoint['critic_state_dict'])