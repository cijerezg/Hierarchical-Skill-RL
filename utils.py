import torch.autograd as autograd
import torch
import copy
import torch.nn as nn
import gym
from collections import OrderedDict
from torch.nn.utils.stateless import functional_call
import numpy as np
import seaborn as sns
from cherry.algorithms import trpo
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import utils
import pdb
import wandb


class hyper_params:
    def __init__(self, args):
        # Batch sizes
        self.meta_batch_size = args.meta_batch_size
        self.batch_size = args.batch_size
        self.vae_batch_size = args.vae_batch_size

        # VAE hierarchy. It should be a dict.
        self.hrchy = self.creating_hierarchy(args)
        self.skill_length = np.prod([lev['length'] for lev in self.hrchy.values()])
        self.z_action = args.z_action
        
        # Learning rates
        self.rl_lr = args.rl_lr
        self.vae_lr = args.vae_lr

        # RL hyperparameters:
        self.discount = args.discount
        self.alpha = args.alpha

        # Env hyperparameters
        self.env_id = args.env_id
        self.action_dim, self.state_dim = self.env_dims(args.env_id)

        # General
        self.device = torch.device(args.device)
        self.hidden_dim = args.hidden_dim  # Hidden dimension for all NNs

    def env_dims(self, env_id):
        env = gym.make(env_id)
        action_dim = env.action_space.shape[0]
        state_dim = env.observation_space.shape[0]
        env.close()
        del env
        return action_dim, state_dim

    def creating_hierarchy(self, args):
        hrchy = {}
        for i in range(args.levels):
            hrchy[i] = {'length': args.level_length, 'z': args.z_vae}

        return hrchy

    
def gradient(loss: torch.tensor,
             params: list,
             name,
             second_order: bool = False,
             ) -> torch.tensor:
    """Compute gradient.

    Compute gradient of loss with respect to parameters.

    Parameters
    ----------
    loss : torch.tensor
        Scalar that depends on params.
    params : list
        Sequence of tensors that the gradient will be computed with
        respect to
    second_order : bool
        Select to compute second or higher order derivatives.

    Returns
    -------
    torch.tensor
        Flattened gradient.

    Examples
    --------
    loss = torch.abs(model(y) - y)
    grad = gradient(loss, model.parameters())

    """
    grad = autograd.grad(loss, params.values(), retain_graph=True,
                         create_graph=second_order, allow_unused=True)

    grad_vector = nn.utils.parameters_to_vector(grad)

    if 'Prior' not in name and 'Action' not in name and 'Seq' not in name:
        wandb.log({f' Grad {name}':
                   wandb.Histogram(nn.utils.parameters_to_vector(grad).cpu())})
        #for g, w_key in zip(grad, params):
        #    wandb.log({f'Grad {name} {w_key}':
        #               wandb.Histogram(nn.utils.parameters_to_vector(g).cpu())})
    return nn.utils.parameters_to_vector(grad)


def params_update_deep(params: list,
                       grad: torch.tensor,
                       lr: float
                       ) -> list:
    """Apply gradient descent update to params.

    It creates a deepcopy of params to save the updated params. This
    is useful for higher order derivatives.

    Parameters
    ----------
    params : list
        Sequences of tensors. These are the base parameters that will
        be updated.
    grad : torch.tensor
        Flattened tensor containing the gradient.
    lr : float
        Learning rate.

    Returns
    -------
    list
        The updated parameters.

    Examples
    --------
    grad = torch.ones(10)
    w, b = torch.rand(5), torch.rand(b)
    params = {'weight': nn.Parameter(w), 'bias': nn.Parameter(b)}
    lr = 0.1
    new_params = params_update_deep(params, grad, lr)

    """
    params_updt = copy.deepcopy(params)
    start, end = 0, 0
    for name, param in params.items():
        start = end
        end = start + param.numel()
        update = grad[start:end].reshape(param.shape)
        params_updt[name] = param - lr * update
    return params_updt

    
def params_update_shalllow(params: list,
                           grad: torch.tensor,
                           lr: float
                           ) -> list:
    """Apply gradient descent update to params.

    It creates rewrites the params to save the updated params. Do not use this
    when computing higher order derivatives.
    is useful for higher order derivatives.

    Parameters
    ----------
    params : list
        Sequences of tensors. These are the base parameters that will
        be updated.
    grad : torch.tensor
        Flattened tensor containing the gradient.
    lr : float
        Learning rate.

    Returns
    -------
    list
        The updated parameters.

    Examples
    --------
    grad = torch.ones(10)
    w, b = torch.rand(5), torch.rand(b)
    params = {'weight': nn.Parameter(w), 'bias': nn.Parameter(b)}
    lr = 0.1
    new_params = params_update_deep(params, grad, lr)

    """
    params_updt = copy.copy(params)
    start, end = 0, 0
    for name, param in params.items():
        start = end
        end = start + param.numel()
        update = grad[start:end].reshape(param.shape)
        params_updt[name] = param - lr * update
    return params_updt


def GD_full_update(params: dict,
                   losses: list,
                   keys: list,
                   lr: float,
                   ) -> list:
    """Compute GD for multiple models.

    Compute and update parameters of models in the params list with
    respect to the losses in the loss list. Do not use for second or
    higher order derivatives.

    Parameters
    ----------
    params : list
        Each element of the dictionary has the parameters of a model.
    losses : list
        Each element of the list is a scalar tensor. It should match
        the order of the params dict, e.g., first element of loss,
        should correspond to first params.
    lr : float
        Learning rate.

    Returns
    -------
    list
        Update dictionary with all parameters.

    """
    for loss, key in zip(losses, keys):
        grad = gradient(loss, params[key], key)
        # print(f'Norm of {key} is {torch.norm(grad)}')
        if key == 'Encoder':
            lr = lr #* 10
        params[key] = params_update_shalllow(params[key], grad, lr)
    return params


def params_extraction(models: list,
                      names: list,
                      pretrained_params,
                      ) -> dict:
    """Get and init params from model to use with functional call.

    The models list contains the pytorch model. The parameters are
    initialized with bias and std 0, and rest with orthogonal init.

    Parameters
    ----------
    models : list
        Each element contains the pytorch model.
    names : list
        Strings that contains the name that will be assigned.

    Returns
    -------
    dict
        Each dictionary contains params ready to use with functional
        call.

    Examples
    --------
    See vae.py for an example.

    """
    params = OrderedDict()
    for model, name_m, pre_params in zip(models, names, pretrained_params):
        par = {}
        if pre_params is None:
            for name, param in model.named_parameters():
                if 'bias' in name:
                    init = torch.nn.init.constant_(param, 0.0)
                elif 'std' in name:
                    init = torch.nn.init.constant_(param, 0.0)
                else:
                    init = torch.nn.init.xavier_normal_(param, gain=1.0)
                par[name] = nn.Parameter(init)
        else:
            for name, param in model.named_parameters():
                init = pre_params[name]
                par[name] = nn.Parameter(init)
        params[name_m] = par
                
    return params


class Video:
    def __init__(self, policy, decoder, max_steps, device, env_id, skill_length):
        self.policy = policy
        self.decoder = decoder
        self.device = device
        self.env = gym.make(env_id)
        self.env._max_episode_steps = max_steps
        self.skill_length = skill_length
        self.max_steps = max_steps

    def saving_video(self, path, params, target, epoch):
        env = gym.wrappers.RecordVideo(self.env, f'{path}/target_speed_{target}')
        state = env.reset()
        for i in range(self.max_steps // self.skill_length):
            with torch.no_grad():
                state_t = torch.from_numpy(state).to(self.device)
                state_t = state_t.view(1, -1)
                z, _, _, _ = functional_call(self.policy,
                                             params['Policy'],
                                             state_t)
                actions = functional_call(self.decoder, params['Decoder'], z)
                actions = actions.cpu().detach().numpy()
            clipped_actions = np.clip(actions, -1, 1)
            for j in range(actions.shape[1]):
                state, rew, done, info = env.step(clipped_actions[:, j, :])
        env.close()

class Video_skill:
    def __init__(self, policy, decoder, max_steps, device, env_id, skill_length):
        self.policy = policy
        self.decoder = decoder
        self.device = device
        self.env = gym.make(env_id)
        self.env._max_episode_steps = max_steps
        self.skill_length = skill_length
        self.max_steps = max_steps

    def saving_video(self, path, params, target, epoch):
        env = gym.wrappers.RecordVideo(self.env, f'{path}/target_speed_{target}')
        state = env.reset()
        for i in range(self.max_steps // self.skill_length):
            with torch.no_grad():
                state_t = torch.from_numpy(state).to(self.device)
                state_t = state_t.view(1, -1)
                z, _, _, _ = functional_call(self.policy,
                                             params['Policy'],
                                             state_t)
                actions = functional_call(self.decoder, params['Decoder'], z)
                actions = actions.cpu().detach().numpy()
            clipped_actions = np.clip(actions, -1, 1)
            for j in range(actions.shape[1]):
                state, rew, done, info = env.step(clipped_actions[:, j, :])
        env.close()
   

class video_from_actions:
    def __init__(self, env_id, max_steps):
        self.env = gym.make(env_id)
        self.env._max_episodes_steps = max_steps

    def saving_video(self, path, epoch, actions):
        env = gym.wrappers.RecordVideo(self.env, f'{path}/epoch_{epoch}')
        _ = env.reset()
        clipped_actions = np.clip(actions, -1, 1)
        for i in range(actions.shape[0]):
            _ = env.step(clipped_actions[i, :])
        env.close()


#vid = video_from_actions('HalfCheetah-v3', 10)

#vid.saving_video('results', 0, np.random.rand(10, 6))


def load_pretrained_models(c):
    params_act_encoder = torch.load('trained_models/ActionEncoder.pt')
    params_act_decoder = torch.load('trained_models/ActionDecoder.pt')
    
    if c.use_pretrained_VAE:
        vae_mod = torch.load(f'VAE_models/mod_{c.levels}_{c.level_length}_{c.z_vae}_{c.vae_lr}.pt')
        pretrained_params = [params_act_encoder, params_act_decoder,
                             *vae_mod.values()]

    else:
        pretrained_params = [params_act_encoder, params_act_decoder]

    return pretrained_params
