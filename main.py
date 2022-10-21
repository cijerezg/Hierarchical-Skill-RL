"""Run everything."""

from models import Policy, Critic, Encoder, Decoder, Skill_prior
from sampler import Sampler
from algos import MSAC
import torch
import numpy as np
import random
from algos import MSAC
from hierarchical_vae import VAE
import os
import itertools
from utils import params_extraction, load_pretrained_models
import pdb
import wandb
import argparse

wandb.login()

sweep_config = {'method': 'grid'}

metric = {'name': 'loss',
          'goal': 'minimize'}

sweep_config['metric'] = metric


parameters_dict = {
    'meta_batch_size': {
        'value': 20},
    'batch_size': {
        'value': 20},
    'vae_batch_size': {
        'value': 512},
    'levels': {
        'values': [1, 2, 3]},
    'level_length': {
        'values': [2, 4, 8, 16]},
    'z_vae': {
        'values': [2, 4, 8]},
    'rl_lr': {
        'value': .005},
    'vae_lr': {
        'values': [0.05, 0.005]},
    'discount': {
        'value':  0.99},
    'alpha': {
        'value': 0.1},
    'env_id': {
        'value': 'HalfCheetah-v3'},
    'device': {
        'value': 'cuda:1'},
    'hidden_dim': {
        'value': 128},
    'epochs': {
        'value': 150},
    'z_action': {
        'value': 2},
    'beta':{
        'value': 0.1},
    'use_recon': {
        'value': True},
    'use_pretrained_VAE': {
        'value': True}}


sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project='Hierarchical_RL')

device = torch.device('cuda:1')

path_to_forward = 'Dataset/data_forward_vel.pt'
path_to_backward = 'Dataset/data_backward_vel.pt'

folder = 'results/Experiment'

def main(config=None):
    with wandb.init(config=config):

        config = wandb.config
        if not os.path.exists(folder):
            os.makedirs(folder)

        assert not (config.level_length == 16 and config.levels > 1), "Configuration was skipped"
        vae = VAE(path_to_forward, path_to_backward, config)
        
        z_dim = vae.hrchy[len(vae.hrchy) - 1]['z']
        policy = Policy(vae.state_dim, z_dim,
                        vae.hidden_dim).to(vae.device)
        critic = Critic(vae.state_dim, z_dim,
                        vae.hidden_dim).to(vae.device)
        policy = policy.double()
        critic = critic.double()

        sampler = Sampler(policy, vae.evaluate_decoder_hrchy,
                          vae.ActionDecoder, config)

        msac = MSAC(sampler, vae, policy, critic, config)

        vae_models = list(vae.models.values())

        models = [vae.ActionEncoder, vae.ActionDecoder, *vae_models,
                  vae.prior, msac.policy, msac.critic, msac.critic]

        vae_names = list(itertools.chain(*list(vae.names.values())))
        names = ['ActionEncoder', 'ActionDecoder', *vae_names,
                 'Prior', 'Policy', 'Critic', 'Target_critic']

        pretrained_params = load_pretrained_models(config)
        pretrained_params.extend([None] * (len(names) - len(pretrained_params)))

        params = params_extraction(models, names, pretrained_params)

        # VAE training warm up
        if not config.use_pretrained_VAE:
            for vae_epoch in range(600):
                for j in range(len(vae.hrchy)):
                    params = vae.train_level(params, config.vae_lr, config.beta,
                                             j, use_recon=config.use_recon)

            trained_vae_models = {key: params[key] for key in params if key in vae_names}
            torch.save(trained_vae_models,
                       f'VAE_models/mod_{config.levels}_{config.level_length}_{config.z_vae}_{config.vae_lr}.pt')

            
        # Main training loop
        for i in range(config.epochs+1):
            tasks = np.random.uniform(0.0, 2.0, (sampler.meta_batch_size,))
            data, params = msac.train_episode(params, tasks, testing=False)
            if i % (config.epochs // 5) == 0:
                mean_total_reward = []
                for idx, test_task in enumerate(sampler.test_tasks):
                    test_task = np.repeat(test_task, sampler.meta_batch_size)
                    d_test, test_params = msac.train_episode(params, test_task,
                                                             testing=True)
                    test_data = msac.test_episode(test_params)
                    mean_reward = np.mean(np.stack(test_data['reward']))
                    pdb.set_trace()
                    task_n = round(test_task[0], 2)
                    wandb.log({f'Task {task_n}': mean_reward})
                    mean_total_reward.append(mean_reward)
                if i == config.epochs:
                    mean_total_reward = np.mean(np.array(mean_total_reward))
                    print(mean_total_reward)
                    wandb.log({'loss': -mean_total_reward})
                        # video.saving_video(folder, test_params, task_n, i)
                        # videoname = 'rl-video-episode-0.mp4'
                        # path_to_video = f'{folder}/target_speed_{task_n}/'
                        # wandb.log({f'Video. Epoch{i}; task {task_n}':
                        #            wandb.Video(f'{path_to_video}{videoname}',
                        #                        fps=20, format='gif')})


wandb.agent(sweep_id, main)
