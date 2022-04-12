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


    train_with_leak = False
    small_leak = False
    train_with_physics_loss = False
    #no_leak_classification = True
    with_sensors = True

    small_demand_variance = False

    train_WGAN = True
    continue_training = True

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

    if with_sensors:
        load_string = load_string + "_sensors"
        save_string = save_string + "_sensors"

    latent_dim = 8
    activation = nn.LeakyReLU()
    transformer = None#transform_data(a=-1, b=1,
                       #      leak=train_with_leak,
                       #      small=small_leak)

    sensors = {'flow_rate_sensors': [3, 7, 16, 19, 26, 29, 32],
               'head_sensors': [3, 8, 17, 19, 26, 27, 31]}

    if with_sensors:
        output_dim = len(sensors['flow_rate_sensors']) + len(sensors['head_sensors'])
    else:
        output_dim = 66
    training_params = {'latent_dim': latent_dim,
                       'n_critic': 3,
                       'gamma': 10,
                       'n_epochs': 10000,
                       'save_string': save_string,
                       'physics_loss': train_with_physics_loss,
                       'device': device}
    optimizer_params = {'learning_rate': 1e-2}
    dataloader_params = {'data_path': data_path,
                         'num_files': 200000,
                         'transformer': transformer,
                         'batch_size': 256,
                         'shuffle': True,
                         'num_workers': 12,
                         'drop_last': True,
                         'sensors': sensors}
    generator_params = {'latent_dim': latent_dim,
                        'par_dim': 35,
                        'output_dim': output_dim,
                        'activation': activation,
                        'n_neurons': [8, 12, 16, 24],
                        'leak': train_with_leak}

    critic_params = {'activation': activation,
                     'n_neurons': [24, 16, 12, 8]}
    if train_with_leak:
        critic_params['input_dim'] = generator_params['output_dim'] + \
                                     generator_params['par_dim']
    else:
        critic_params['input_dim'] = generator_params['output_dim']

    if train_with_physics_loss:
        critic_params['input_dim'] += 32

    generator = GAN_models.Generator(**generator_params).to(device)
    critic = GAN_models.Critic(**critic_params).to(device)

    dataloader = get_dataloader(**dataloader_params)
    lol, _ = dataloader.dataset[0]

    '''
    data_dict = nx.read_gpickle(data_path + str(0))

    flow_rate = torch.tensor(data_dict['flow_rate'].values)[0]
    head = torch.tensor(data_dict['head'].values)[0]
    demand = torch.tensor(data_dict['demand'].values)[0]

    data = torch.cat([flow_rate, head], dim=0)

    pars = torch.zeros([35, ])
    pars[0] = torch.tensor(data_dict['leak']['demand'])
    pars[data_dict['leak']['pipe']] = 1
    data = torch.cat([data, pars], dim=0)

    pdb.set_trace()
    data = transformer.min_max_transform(data)  
    '''
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
                                     dtype=torch.get_default_dtype())

        id = 50

        z = torch.randn(100, latent_dim).to(device)
        out = generator(z)
        out = transformer.min_max_inverse_transform(out.cpu())
        demand_pred = torch.matmul(-incidence_mat.T, out[:,0:34].T).detach().cpu()

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
                           **training_params, transformer=transformer)


        generator_loss, critic_loss, gradient_penalty = trainer.train(
                data_loader=dataloader)