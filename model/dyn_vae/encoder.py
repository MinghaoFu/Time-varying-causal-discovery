import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNNEncoder(nn.Module):
    def __init__(self,
                 args,
                 ):
        super(RNNEncoder, self).__init__()

        self.args = args
        self.reparameterise = self._sample_gaussian
        self.value_expand = FeatureExtractor(1, args.encoder_hidden_dim, F.relu)
        self.prior_time_encoder = FeatureExtractor(args.encoder_hidden_dim, args.encoder_gru_hidden_size, F.relu)
        self.state_encoder = FeatureExtractor(args.encoder_hidden_dim, args.state_embedding_dim, F.relu)
        self.time_encoder = FeatureExtractor(1, args.time_embedding_dim, F.relu)
        curr_input_dim = args.state_embedding_dim 

        self.fc_before_gru = nn.ModuleList([])
        for i in range(len(args.encoder_layers_before_gru)):
            self.fc_before_gru.append(nn.Linear(curr_input_dim, args.encoder_layers_before_gru[i]))
            curr_input_dim = args.encoder_layers_before_gru[i]

        self.gru = nn.GRU(input_size=curr_input_dim,
                          hidden_size=args.encoder_gru_hidden_size,
                          num_layers=1,
                          )
        
        self.transformer = nn.Transformer(
            d_model=curr_input_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            )

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        curr_input_dim = args.encoder_gru_hidden_size
        self.fc_after_gru = nn.ModuleList([])
        for i in range(len(args.encoder_layers_after_gru)):
            self.fc_after_gru.append(nn.Linear(curr_input_dim, args.encoder_layers_after_gru[i]))
            curr_input_dim = args.encoder_layers_after_gru[i]

        self.fc_mu = nn.Linear(curr_input_dim, args.d_L)
        self.fc_logvar = nn.Linear(curr_input_dim, args.d_L)
        
        self.hidden_state = None

    def _sample_gaussian(self, mu, logvar, num=None):
        if num is None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            raise NotImplementedError
            std = torch.exp(0.5 * logvar).repeat(num, 1)
            eps = torch.randn_like(std)
            mu = mu.repeat(num, 1)
            return eps.mul(std).add_(mu)

    def reset_hidden(self, hidden_state, done):
        if hidden_state.dim() != done.dim():
            if done.dim() == 2:
                done = done.unsqueeze(0)
            elif done.dim() == 1:
                done = done.unsqueeze(0).unsqueeze(2)
        hidden_state = hidden_state * (1 - done)
        return hidden_state

    def prior(self, batch_size, t_state, sample=True):
        """
            t_state: [1 * batch_size * encoder_hidden_dim]
        """
        hidden_state = self.prior_time_encoder(t_state)
        #hidden_state = check_tensor(torch.zeros((1, batch_size, self.args.encoder_gru_hidden_size), requires_grad=True))
        h = hidden_state

        for i in range(len(self.fc_after_gru)):
            h = F.relu(self.fc_after_gru[i](h))

        latent_mean = self.fc_mu(h)
        latent_logvar = self.fc_logvar(h)
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        return latent_sample, latent_mean, latent_logvar, hidden_state

    def forward(self, X, T, return_prior=True, sample=True, detach_every=None):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        """
        XT = torch.cat((X, T), dim=1).unsqueeze(-1)
        states = self.value_expand(XT).transpose(0, 1)
        
        if return_prior:
            prior_sample, prior_mean, prior_logvar, prior_hidden_state = self.prior(X.shape[1], states[-1:, :, :])
            hidden_state = prior_hidden_state.clone()
        
        h = self.state_encoder(states)

        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_gru[i](h))
        if detach_every is None:
            output, _ = self.gru(h, hidden_state)
        else:
            output = []
            for i in range(int(np.ceil(h.shape[0] / detach_every))):
                curr_input = h[i * detach_every:i * detach_every + detach_every]
                curr_output, hidden_state = self.gru(curr_input, hidden_state)
                output.append(curr_output)
                hidden_state = hidden_state.detach()
            output = torch.cat(output, dim=0)
        gru_h = output.clone()
        
        self.hidden_state = hidden_state
        
        for i in range(len(self.fc_after_gru)):
            gru_h = F.relu(self.fc_after_gru[i](gru_h))

        gru_h = gru_h.mean(dim=0, keepdim=True)

        latent_mean = self.fc_mu(gru_h)
        latent_logvar = self.fc_logvar(gru_h)
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        if return_prior:
            latent_sample = torch.cat((prior_sample, latent_sample))
            latent_mean = torch.cat((prior_mean, latent_mean))
            latent_logvar = torch.cat((prior_logvar, latent_logvar))
            output = torch.cat((prior_hidden_state, output))
            
        if latent_mean.shape[0] == 1:
            latent_sample, latent_mean, latent_logvar = latent_sample[0], latent_mean[0], latent_logvar[0]
        return latent_sample, latent_mean, latent_logvar, output
