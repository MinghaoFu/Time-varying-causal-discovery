
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import math
from utils import check_tensor, top_k_abs_tensor
from torch.nn.utils import spectral_norm
from snake.activations import Snake

class TApproximator(nn.Module):
    def __init__(self, hid_dim, pi, periodic_ratio=0.2):
        super(TApproximator, self).__init__()
        self.pi = pi
        self.n_periodic_node = int(hid_dim * periodic_ratio)
        self.fc1 = nn.Linear(1, hid_dim - self.n_periodic_node)
        if self.n_periodic_node != 0:
            self.fc1_ = nn.Linear(1, self.n_periodic_node)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, 1)
        self.snake = Snake(hid_dim, 10)
        self.sigmoid = nn.Sigmoid()
        self.beta = nn.Parameter(torch.Tensor([1.0]))
        self.bias = nn.Parameter(torch.Tensor([1.0]))
        
        # init.uniform_(self.fc1.bias, -5000, 5000)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.sigmoid(x1)
        if self.n_periodic_node != 0:
            x2 = self.fc1_(x)
            x2 = torch.cos(self.beta * torch.pi * x2 / self.pi + self.bias)
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.sigmoid(x)
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

class GolemModel(nn.Module):
    def __init__(self, args, d, device, in_dim=1, equal_variances=True,
                 seed=1, B_init=None):
        super(GolemModel, self).__init__()
        # self.n = n
        self.d = d
        self.seed = seed
        self.loss = args.loss
        self.batch_size = args.batch_size
        self.equal_variances = equal_variances
        self.B_init = B_init
        self.in_dim = in_dim
        self.embedding_dim = args.embedding_dim
        self.num = args.num
        self.distance = args.distance
        self.tol = args.tol
        self.device = device

        self.B_lags = []
        self.lag = args.lag
        
        self.gradient = []
        self.Bs = np.empty((0, d, d))
        self.TApproximators = nn.ModuleList([TApproximator(64, args.pi, periodic_ratio=0.2) for _ in range(self.d ** 2 - self.d - (self.d - self.distance) * (self.d - self.distance - 1))])

    def decompose_t_batch(self, t_batch):
        a_batch = t_batch // 100
        b_batch = t_batch % 100
        return a_batch, b_batch
    
    def apply_spectral_norm(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                spectral_norm(module)
        
    def generate_B(self, T):
        #T_embed = (1 / self.alpha) * torch.cos(self.beta * T + self.bias)
        T_embed = T
        B = []
        for layer in self.TApproximators:
            B_i = layer(T.unsqueeze(1))
            B_i_sparse = B_i.masked_fill_(torch.abs(B_i) < self.tol, 0)
            B.append(B_i_sparse)
        B = torch.cat(B, dim=1)
    
        B = self.reshape_B(B)
        return B, T_embed
        
    def _preprocess(self, B):
        B = B.clone()
        B_shape = B.shape
        if len(B_shape) == 3:  # Check if B is a batch of matrices
            for i in range(B_shape[0]):  # Iterate over each matrix in the batch
                B[i].fill_diagonal_(0)
        else:
            print("Input tensor is not a batch of matrices.")
            B.data.fill_diagonal_(0)
        return B

    
    def forward_latent(self, X, T, B_init=None, init_f=False, B_label = None):
        B, T_embed = self.generate_B(T)
        self.Bs = np.concatenate((self.Bs, B.detach().cpu().numpy()), axis=0)
        
        return B
        
    def forward(self, T, B_label=None):
        B, T_embed = self.generate_B(T)
        if self.training is False:
            self.Bs = np.concatenate((self.Bs, B.detach().cpu().numpy()), axis=0)
        else:
            self.Bs = self.Bs[:0]
        
        return B
        # B, T_embed = self.generate_B(T)
        # if init_f:
        #     if B_label is not None:
        #         label = B_label.reshape(B_label.shape[0], self.d, self.d)
        #     else:
        #         label = check_tensor(B_init).repeat(B.shape[0], 1, 1)
            
        # self.Bs.append(B.detach().cpu().numpy())
        
        # if init_f:
        #     losses = {'total_loss': torch.nn.functional.mse_loss(B, label)}
        #     return losses, B 
        # else:
        #     losses = self.compute_loss(X, B, T_embed) 
        #     return losses
    
    def reshape_B(self, B):
        B_zeros = check_tensor(torch.zeros(B.shape[0], self.d, self.d))
        idx = 0
        for i in range(self.d ** 2):
            row = i // self.d
            col = i % self.d
            if -self.distance <= col - row <= self.distance and row != col:
                B_zeros[:, row, col] = B[:, idx]
                idx += 1
            else:
                continue
        return B_zeros
    
    def mask_B(self, B, fix: bool):
        if fix:
            mask = check_tensor(torch.zeros(B.shape[0], self.d, self.d))

            indices = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 2), (4, 3)]
            for i, j in indices:
                mask[:, i, j] = 1.0
            masked_data = B * mask
            return masked_data
        else:
            l_mask = self.d - self.distance - 1
            mask_upper = torch.triu(torch.zeros((self.d, self.d)), diagonal=1)
            mask_upper[:l_mask, -l_mask:] = torch.triu(torch.ones((l_mask, l_mask)), diagonal=0)

            mask_lower = torch.tril(torch.zeros((self.d, self.d)), diagonal=-1)
            mask_lower[-l_mask:, :l_mask] = torch.tril(torch.ones((l_mask, l_mask)), diagonal=0)
            mask = mask_upper + mask_lower
            mask = mask.expand(self.batch_size, self.d, self.d)
            B = B * check_tensor(1 - mask)
            for i in range(self.batch_size):
                B[i] = top_k_abs_tensor(B[i], 6)
            return B

    def step(self, X):
        self.optimizer.zero_grad()
        loss = self.forward(X)
        loss.backward()
        self.optimizer.step()
        return loss.item()