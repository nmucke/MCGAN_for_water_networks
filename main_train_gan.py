import pdb
import torch.nn as nn
import torch
from data_handling.gan_dataloaders import get_dataloader
import models.GAN_models as GAN_models
from utils.load_checkpoint import load_checkpoint
from transforms.transform_data import transform_data
from utils.seed_everything import seed_everything
from training.training_GAN import TrainGAN
from utils.graph_utils import get_incidence_mat
import networkx as nx
torch.set_default_dtype(torch.float32)
import pickle
import os
import wntr
import matplotlib.pyplot as plt

if __name__ == "__main__":

    seed_everything()

    cuda = False
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Training GAN on {device}')


    train_with_leak = True
    small_leak = False
    train_with_physics_loss = 1e-4

    small_demand_variance = False

    train_WGAN = True
    continue_training = False

    if train_with_leak:
        if small_leak:
            data_path = 'data/training_data_with_leak_small/network_'
            load_string = 'model_weights/GAN_small_leak'
            save_string = 'model_weights/GAN_small_leak'
        else:
            if small_demand_variance:
                data_path = 'data/training_data_with_leak_small_demand_variance/network_'
                load_string = 'model_weights/GAN_leak_small_demand_variance'
                save_string = 'model_weights/GAN_leak_small_demand_variance'
            else:
                data_path = 'data/training_data_with_leak/network_'
                load_string = 'model_weights/GAN_leak'
                save_string = 'model_weights/GAN_leak'
            if train_with_physics_loss:
                data_path = 'data/training_data_with_leak/network_'
                load_string = 'model_weights/GAN_leak_and_pysics_loss'
                save_string = 'model_weights/GAN_leak_and_pysics_loss'

    else:
        if small_demand_variance:
            data_path = 'data/training_data_no_leak_small_demand_variance/network_'
            load_string = 'model_weights/GAN_no_leak_small_demand_variance'
            save_string = 'model_weights/GAN_no_leak_small_demand_variance'
        else:
            data_path = 'data/training_data_no_leak/network_'
            load_string = 'model_weights/GAN_no_leak'
            save_string = 'model_weights/GAN_no_leak'


    latent_dim = 16
    activation = nn.LeakyReLU()
    transformer = None#transform_data(a=-1, b=1,
                      #           leak=train_with_leak,
                      #           small=small_leak)

    training_params = {'latent_dim': latent_dim,
                       'n_critic': 2,
                       'gamma': 10,
                       'n_epochs': 1000,
                       'save_string': save_string,
                       'physics_loss': train_with_physics_loss,
                       'device': device}
    optimizer_params = {'learning_rate': 1e-4}
    dataloader_params = {'data_path': data_path,
                         'num_files': 200000,
                         'transformer': transformer,
                         'batch_size': 256,
                         'shuffle': True,
                         'num_workers': 12,
                         'drop_last': True}
    generator_params = {'latent_dim': latent_dim,
                        'par_dim': 35,
                        'output_dim': 66,
                        'activation': activation,
                        'n_neurons': [16, 32, 48, 64],
                        'leak': train_with_leak}

    critic_params = {'activation': activation,
                     'n_neurons': [96, 80, 64, 48, 32, 16]}
    if train_with_leak:
        critic_params['input_dim'] = generator_params['output_dim'] + \
                                     generator_params['par_dim']
    else:
        critic_params['input_dim'] = generator_params['output_dim']

    generator = GAN_models.Generator(**generator_params).to(device)
    critic = GAN_models.Critic(**critic_params).to(device)

    dataloader = get_dataloader(**dataloader_params)

    if train_WGAN:

        generator_optimizer = torch.optim.RMSprop(generator.parameters(),
                                                  lr=optimizer_params['learning_rate'])
        critic_optimizer = torch.optim.RMSprop(critic.parameters(),
                                               lr=optimizer_params['learning_rate'])

        if continue_training:
            load_checkpoint(load_string, generator, critic,
                            generator_optimizer, critic_optimizer)
        '''
        inputfiles_folder_name = 'Input_files_EPANET'
        filename = 'Hanoi_base_demand.inp'
        path_file = os.path.join(inputfiles_folder_name, filename)

        # Reading the input file into EPANET
        inp_file = path_file
        wn = wntr.network.WaterNetworkModel(inp_file)
        incidence_mat = torch.tensor(get_incidence_mat(wn),
                                     dtype=torch.get_default_dtype(),
                                     device=device)

        z = torch.randn(10, latent_dim).to(device)
        out = generator(z)
        demand_pred = torch.matmul(-incidence_mat.T, out[:,0:34].T).detach().cpu()

        id = 1

        print(demand_pred.sum(dim=0))
        print(out[:, 66])

        leak_location = torch.argmax(out[:, -34:], dim=1).detach().cpu()
        print(leak_location)

        plt.figure()
        plt.plot(demand_pred[:, id])
        plt.axvline(x=leak_location[id])
        plt.show()
        pdb.set_trace()
        '''


        trainer = TrainGAN(generator=generator,
                           critic=critic,
                           generator_optimizer=generator_optimizer,
                           critic_optimizer=critic_optimizer,
                           **training_params)


        generator_loss, critic_loss, gradient_penalty = trainer.train(
                data_loader=dataloader)