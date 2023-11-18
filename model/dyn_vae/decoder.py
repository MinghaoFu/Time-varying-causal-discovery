import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import check_tensor

from .base import *
        
class TimeLagTransitionDecoder(nn.Module):
    def __init__(self, args, ):
        super(TimeLagTransitionDecoder, self).__init__()

        self.args = args
        self.curr_adj_encoder = MLP(1, args.d_L * args.d_X, args.d_L, args.n_mlp_layers, 'lrelu')
        self.pre_adj_encoder = MLP(1, args.d_L * args.d_X, args.d_L, args.n_mlp_layers, 'lrelu')
        self.latent_mean_out = nn.Sequential(
            FeatureExtractor(args.d_L, 128, F.relu),
            nn.Linear(128, args.d_L)
        )
        self.latent_logvar_out = nn.Sequential(
            FeatureExtractor(args.d_L, 128, F.relu),
            nn.Linear(128, args.d_L)
        )
        
        
        self.L2M_layers = nn.ModuleList([])
        for i in range(args.d_X):
            L2m_layer = nn.Sequential()
            curr_input_dim = 4 * args.d_L 
            for i in range(len(args.decoder_layer_dims)):
                L2m_layer.append(nn.Linear(curr_input_dim, args.decoder_layer_dims[i]))
                if i != len(args.decoder_layer_dims) - 1:
                    L2m_layer.append(nn.ReLU())
                curr_input_dim = args.decoder_layer_dims[i]
            L2m_layer.append(nn.Linear(curr_input_dim, 1))
            self.L2M_layers.append(L2m_layer)
            
        self.fc_out = nn.Linear(curr_input_dim, args.d_X)
    
    def decode(self, latent_mean, latent_logvar, C):  
        
        latent_mean = torch.einsum('ijk,lik->lijk', C, latent_mean).permute(1, 2, 3, 0).reshape(latent_mean.shape[1], self.args.d_X, -1) 
        latent_logvar = torch.einsum('ijk,lik->lijk', C, latent_logvar).permute(1, 2, 3, 0).reshape(latent_logvar.shape[1], self.args.d_X, -1)
        h = torch.cat((latent_mean, latent_logvar), dim=-1)
        
        hs = []
        for i, layer in enumerate(self.L2M_layers):
            hs.append(F.relu(layer(h[:, i, :])))
        return torch.cat(hs, dim=1)
    
    def forward(self, latent_mean, latent_logvar, T):

        Ct = self.curr_adj_encoder(T).reshape(-1, self.args.d_X, self.args.d_L)
        Ct_1 = self.pre_adj_encoder(T).reshape(-1, self.args.d_X, self.args.d_L)
        bin_thres = 0.1
        Ct[Ct < bin_thres] = 0
        Ct[Ct >= bin_thres] = 1
        
        # latent_mean = latent_mean[0]
        # prior_mean = latent_mean[1] 
        # latent_logvar = latent_logvar[0]
        # prior_logvar = latent_logvar[1]
        latent_zeros = check_tensor(torch.zeros_like(latent_mean[:, 0:1, :]))
        latent_mean_prev = torch.cat((latent_zeros, latent_mean[:, :-1, :]), dim=1)
        latent_logvar_prev = torch.cat((latent_zeros, latent_logvar[:, :-1, :]), dim=1)
                
        # latent_mean = latent_mean * Ct[:, None, :]
        # latent_logvar = latent_logvar * Ct[:, None, :]
        # latent_mean_prev = latent_mean_prev * Ct_1
        # latent_logvar_prev = latent_logvar_prev * Ct_1
        
        latent_mean_pred = self.latent_mean_out(latent_mean)
        latent_logvar_pred = self.latent_logvar_out(latent_mean)
        recon = self.decode(latent_mean, latent_logvar, Ct)
        pred = self.decode(latent_mean_prev, latent_logvar_prev, Ct_1)

        latent_mean_pred = self.latent_mean_out(latent_mean)
        latent_logvar_pred = self.latent_logvar_out(latent_logvar)
    
        return recon, pred, latent_mean_pred, latent_logvar_pred, Ct, Ct_1
    
class InstantaneousDecoder(nn.Module):
    def __init__(self, args, ):
        super(InstantaneousDecoder, self).__init__()
        self.args = args
        self.curr_adj_encoder = MLP(1, args.d_L * args.d_X, args.d_L, args.n_mlp_layers, 'lrelu')
        self.pre_adj_encoder = MLP(1, args.d_L * args.d_X, args.d_L, args.n_mlp_layers, 'lrelu')
        self.latent_mean_out = nn.Sequential(
            FeatureExtractor(args.d_L, 128, F.relu),
            nn.Linear(128, args.d_L)
        )
        self.latent_logvar_out = nn.Sequential(
            FeatureExtractor(args.d_L, 128, F.relu),
            nn.Linear(128, args.d_L)
        )
        
        self.L2M_layers = nn.ModuleList([])
        for i in range(args.d_X):
            L2m_layer = nn.Sequential()
            curr_input_dim = 4 * args.d_L 
            for i in range(len(args.decoder_layer_dims)):
                L2m_layer.append(nn.Linear(curr_input_dim, args.decoder_layer_dims[i]))
                if i != len(args.decoder_layer_dims) - 1:
                    L2m_layer.append(nn.ReLU())
                curr_input_dim = args.decoder_layer_dims[i]
            L2m_layer.append(nn.Linear(curr_input_dim, 1))
            self.L2M_layers.append(L2m_layer)
            
        self.fc_out = nn.Linear(curr_input_dim, args.d_X)
    
    def decode(self, latent_mean, latent_logvar, C):  
        
        latent_mean = torch.einsum('ijk,lik->lijk', C, latent_mean).permute(1, 2, 3, 0).reshape(latent_mean.shape[1], self.args.d_X, -1) 
        latent_logvar = torch.einsum('ijk,lik->lijk', C, latent_logvar).permute(1, 2, 3, 0).reshape(latent_logvar.shape[1], self.args.d_X, -1)
        h = torch.cat((latent_mean, latent_logvar), dim=-1)
        
        hs = []
        for i, layer in enumerate(self.L2M_layers):
            hs.append(F.relu(layer(h[:, i, :])))
        return torch.cat(hs, dim=1)
    
    def forward(self, latent_mean, latent_logvar, T):

        Ct = self.curr_adj_encoder(T).reshape(-1, self.args.d_X, self.args.d_L)
        Ct_1 = self.pre_adj_encoder(T).reshape(-1, self.args.d_X, self.args.d_L)
        bin_thres = 0.1
        Ct[Ct < bin_thres] = 0
        Ct[Ct >= bin_thres] = 1
        
        latent_zeros = check_tensor(torch.zeros_like(latent_mean[:, 0:1, :]))
        latent_mean_prev = torch.cat((latent_zeros, latent_mean[:, :-1, :]), dim=1)
        latent_logvar_prev = torch.cat((latent_zeros, latent_logvar[:, :-1, :]), dim=1)
        
        latent_mean_pred = self.latent_mean_out(latent_mean)
        latent_logvar_pred = self.latent_logvar_out(latent_mean)
        recon = self.decode(latent_mean, latent_logvar, Ct)
        pred = self.decode(latent_mean_prev, latent_logvar_prev, Ct_1)

        latent_mean_pred = self.latent_mean_out(latent_mean)
        latent_logvar_pred = self.latent_logvar_out(latent_logvar)
    
        return recon, pred, latent_mean_pred, latent_logvar_pred, Ct, Ct_1