"""Implementation of algorithms."""

import torch.autograd as autograd
import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from utils import GD_full_update
import numpy as np
import copy
import torch.nn as nn
import seaborn as sns
from torch.nn.utils.stateless import functional_call
from cherry.algorithms import trpo
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils import hyper_params
import wandb
import utils
import pdb

sns.set_theme()


class MSAC(hyper_params):
    def __init__(self,
                 sampler,
                 vae,
                 policy,
                 critic,
                 args,):
        super().__init__(args)
        self.sampler = sampler
        self.critic = critic
        self.policy = policy
        self.vae = vae
        self.losses_list = {'policy_loss': [], 'critic_loss': []}

    def iteration(self, params, data, done, tasks, testing=False):
        data, done = self.sampler.skill_length_step(params, data, tasks, done)
        policy_loss, critic_loss = self.losses(params, data)
        losses = [policy_loss, critic_loss]
        params = GD_full_update(params, losses, ['Policy', 'Critic'], 0.01)
        self.losses_list['policy_loss'].append(policy_loss.detach())
        self.losses_list['critic_loss'].append(critic_loss.detach())
        if ~testing:
            wandb.log({'policy_loss': policy_loss})
            wandb.log({'critic_loss': critic_loss})
        return data, params, done
        
    def train_episode(self, params, tasks, testing):
        data = {var: [] for var in self.sampler.variables}
        done = np.array([False, False])
        while ~done.all():
            data, params, done = self.iteration(params, data, done, tasks,
                                                testing=testing)
        return data, params

    def losses(self, params, data):
        state_t = torch.from_numpy(data['state'][-1]).to(self.device)
        next_state_t = torch.from_numpy(data['next_state'][-1]).to(self.device)
        reward_t = torch.from_numpy(data['reward'][-1]).to(self.device)
        
        target_critic_arg = torch.cat((next_state_t, data['next_z'][-1]), dim=1)
        q_target = functional_call(self.critic, params['Target_critic'],
                                   target_critic_arg)

        critic_arg = torch.cat((state_t, data['next_z'][-1]), dim=1)
        q = functional_call(self.critic, params['Critic'], critic_arg)
        
        with torch.no_grad():
            next_density_prior = functional_call(self.vae.prior, params['Prior'],
                                                 next_state_t)
            density_prior = functional_call(self.vae.prior, params['Prior'],
                                            state_t)

        next_kl_pi_prior = kl_divergence(data['next_density'][-1], next_density_prior)
        kl_pi_prior = kl_divergence(data['density'][-1], density_prior)
        q_target = reward_t - self.discount * \
            (q_target.squeeze() - self.alpha * 
             torch.mean(next_kl_pi_prior, axis=1))

        policy_loss = torch.mean(q.squeeze() - self.alpha * 
                                 torch.mean(kl_pi_prior, axis=1))
        critic_loss = F.mse_loss(q.squeeze(), q_target)
        return policy_loss, critic_loss

    def test_episode(self, params):
        return self.sampler.full_episode(params)
    

