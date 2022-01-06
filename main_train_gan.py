import pdb
import torch.nn as nn
import torch
from data_handling.gan_dataloaders import get_dataloader
import models.GAN_models as GAN_models
from utils.load_checkpoint import load_checkpoint
from transforms.transform_data import transform_data
from utils.seed_everything import seed_everything
from training.training_GAN import TrainGAN

torch.set_default_dtype(torch.float32)

if __name__ == "__main__":

    seed_everything()

    cuda = True
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Training GAN on {device}')


    train_with_leak = True
    small_leak = False

    if train_with_leak:
        if small_leak:
            data_path = 'data/training_data_with_leak_small/network_'
        else:
            data_path = 'data/training_data_with_leak/network_'
    else:
        data_path = 'data/training_data_no_leak/network_'

    train_WGAN = True
    continue_training = False
    load_string = 'model_weights/GAN_leak'
    save_string = 'model_weights/GAN_leak'

    latent_dim = 32
    activation = nn.LeakyReLU()
    transformer = transform_data(a=-1, b=1,
                                 leak=train_with_leak,
                                 small=small_leak)

    training_params = {'latent_dim': latent_dim,
                       'n_critic': 3,
                       'gamma': 10,
                       'n_epochs': 10000,
                       'save_string': save_string,
                       'device': device}
    optimizer_params = {'learning_rate': 1e-4}
    dataloader_params = {'data_path': data_path,
                         'num_files': 100000,
                         'transformer': transformer,
                         'batch_size': 512,
                         'shuffle': True,
                         'num_workers': 24,
                         'drop_last': True}
    generator_params = {'latent_dim': latent_dim,
                        'par_dim': 33,
                        'output_dim': 66,
                        'activation': activation,
                        'n_neurons': [16, 32, 64],
                        'leak': train_with_leak}

    critic_params = {'activation': activation,
                     'n_neurons': [64, 32, 16]}
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

        trainer = TrainGAN(generator=generator,
                           critic=critic,
                           generator_optimizer=generator_optimizer,
                           critic_optimizer=critic_optimizer,
                           **training_params)


        generator_loss, critic_loss, gradient_penalty = trainer.train(
                data_loader=dataloader)