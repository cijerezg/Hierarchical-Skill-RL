"""VAE functions."""

from models import ActionEncoder, ActionDecoder, Skill_prior
from models import EncoderTransformer, DecoderTransformer
from utils import hyper_params
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn.utils.stateless import functional_call
import torch
import wandb
from utils import GD_full_update, params_extraction
import pdb



class VAE(hyper_params):
    def __init__(self, path_to_forward, path_to_backward, config):
        super().__init__(config)

        self.dataset = self.prepare_dataset(path_to_forward, path_to_backward)
        dset_train = Drivedata(self.dataset)

        self.loader = DataLoader(dset_train, shuffle=True, num_workers=8,
                                 batch_size=self.vae_batch_size)
        self.test_loader = DataLoader(dset_train, shuffle=False, num_workers=8,
                                      batch_size=self.dataset['actions'].shape[0])

        self.models = {}
        self.names = {}

        self.ActionEncoder = ActionEncoder(self.action_dim,
                                           self.z_action,
                                           self.hidden_dim,
                                           1).to(self.device)

        self.ActionDecoder = ActionDecoder(self.z_action,
                                           self.action_dim,
                                           self.hidden_dim,
                                           1).to(self.device)

        self.prior = Skill_prior(self.state_dim,
                                 self.hrchy[len(self.hrchy) - 1]['z'],
                                 self.hidden_dim).to(self.device)

        self.ActionEncoder = self.ActionEncoder.double()
        self.ActionDecoder = self.ActionDecoder.double()
        self.prior = self.prior.double()
        
        for idx, level in enumerate(self.hrchy):
            if idx == 0:
                old_z = self.z_action
            else:
                old_z = list(self.hrchy.values())[idx-1]['z']
                
            self.encoder = EncoderTransformer(
                old_z,
                self.hrchy[level]['z'],
                self.hidden_dim,
                1).to(self.device)

            self.decoder = DecoderTransformer(
                self.hrchy[level]['z'],
                old_z,
                self.hidden_dim,
                1,
                self.hrchy[level]['length']).to(self.device)

            self.models[f'SeqEncoder{level}'] = self.encoder.double()
            self.models[f'SeqDecoder{level}'] = self.decoder.double()

            self.names[idx] = [f'SeqEncoder{level}', f'SeqDecoder{level}']


    def train_level(self, params, lr, beta, level, use_recon=True):
        for action, obs in self.loader:
            action, obs = action.to(self.device), obs.to(self.device)
            recon_loss, kl_loss, kl_prior = self.loss(action, obs, params,
                                                      level,
                                                      use_recon=use_recon)
                        
            loss = recon_loss + beta * kl_loss
            if level != (len(self.hrchy) - 1):
                losses = [loss, recon_loss] 
                params_names = self.names[level]
            elif level == (len(self.hrchy) - 1):
                losses = [loss, recon_loss, kl_prior]
                params_names = self.names[level]
                params_names.append('Prior')
            params = GD_full_update(params, losses, params_names, lr)
        wandb.log({f'recon_loss_level_{level}': recon_loss})
        wandb.log({f'kl_loss_level_{level}': kl_loss})
        wandb.log({f'kl_prior_level_{level}': kl_prior})
        wandb.log({'loss': recon_loss})
        wandb.log({'kl_loss': kl_loss})

        return params
    

    def loss(self, action, obs, params, level, use_recon=True):
        with torch.no_grad():
            z_act, _, _, _ = functional_call(self.ActionEncoder,
                                             params['ActionEncoder'],
                                             action)
        z_seq, pdf = self.evaluate_encoder_hrchy(z_act, params, level)
        rec = self.evaluate_decoder_hrchy(z_seq, params, level)
        try:
            if use_recon:
                z_rec = functional_call(self.ActionDecoder,
                                        params['ActionDecoder'],
                                        rec)
                rec_loss = F.mse_loss(action, z_rec)
            else:
                rec_loss = F.mse_loss(rec, z_act)

            N = Normal(0, 1)
            kl_loss = torch.mean(kl_divergence(pdf, N))
            if level == len(self.hrchy) - 1:
                # Problem is with kl prior. Shapes are not matching.
                kl_prior = self.train_prior(params, pdf, obs)
            else:
                kl_prior = None
        except RuntimeError:
            pdb.set_trace()
            print('wait')

        return rec_loss, kl_loss, kl_prior
            
    def evaluate_encoder_hrchy(self, z, params, level):
        for i in range(level+1):
            z = z.reshape(-1, self.hrchy[level]['length'], z.shape[-1])
            try:
                z, pdf, _, _ = functional_call(self.models[f'SeqEncoder{i}'],
                                               params[f'SeqEncoder{i}'], z)
            except RuntimeError:
                pdb.set_trace()
                print('a')

        return z, pdf

    def evaluate_decoder_hrchy(self, rec, params, level):
        for i in range(level, -1, -1):
            try:
                rec = functional_call(self.models[f'SeqDecoder{i}'],
                                      params[f'SeqDecoder{i}'], rec)
            except RuntimeError:
                pdb.set_trace()

            rec = rec.reshape(-1, rec.shape[-1])

        return rec

    def train_prior(self, params, pdf, obs):
        index = obs.shape[0] // pdf.loc.shape[0]
        obs = obs[::index, :]
        prior = functional_call(self.prior, params['Prior'], obs)
        return torch.mean(kl_divergence(pdf, prior))

    def prepare_dataset(self, path_forward, path_backward):
        data_f = torch.load(path_forward)
        data_b = torch.load(path_backward)
        dataset_seqs = {}
        
        for key in data_f:
            # Select data_f[key][-1]. The -1 is because the data was saved
            # two times during training. Use -1.
            seqs = torch.cat((data_f[key][-1], data_b[key][-1]), dim=0)
            seqs = seqs[:, :960, :]
            seqs = seqs.reshape(-1, seqs.shape[2])
            # shape = [self.hrchy[lev]['length'] for lev in self.hrchy if 'action' not in lev]
            # shape.insert(0, -1)
            # shape.append(seqs[-1])
            # shape = tuple(shape)
            # seqs = seqs.reshape(shape)
            dataset_seqs[key] = seqs.double()

        return dataset_seqs

    
class Drivedata(Dataset):
    """Dataset loader."""
    def __init__(self, dataset, transform=None):
        self.xs = dataset['actions']
        self.ys = dataset['states']

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]

    def __len__(self):
        return len(self.xs)
