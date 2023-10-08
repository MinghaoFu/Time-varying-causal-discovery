import torch
import logging

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
        self.device = device

        self.B_lags = []
        self.lag = args.lag
        
        self.gradient = []
        self.Bs = []

        hid_dim = 128 * self.embedding_dim
        # rnn
        self.rnn = nn.RNN(self.embedding_dim, 2 * self.embedding_dim, 1, batch_first=True)
        self.fc = nn.Linear(in_features=2 * self.embedding_dim, out_features=self.d * self.d)
        # periodic
        self.alpha = nn.Parameter(torch.tensor([0.1]))
        self.beta = nn.Parameter(torch.tensor([100.0]))
        self.bias = nn.Parameter(torch.tensor([1.0]))
        
        self.TApproximators = nn.ModuleList([TApproximator(64, args.pi, periodic_ratio=0.2).cuda() for _ in range(self.d ** 2 - (self.d - self.distance) * (self.d - self.distance - 1))])
        
        self.linear1 = nn.Linear(in_features=2, out_features=hid_dim)
        self.linear2 = nn.Linear(in_features=hid_dim, out_features=hid_dim)
        self.linear3 = nn.Linear(in_features=hid_dim, out_features=self.d * self.d)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.snake = Snake(hid_dim, 25)
        if args.spectral_norm:
            self.apply_spectral_norm()

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
            B.append(B_i)
        B = torch.cat(B, dim=1)
        # h0 = self.linear1(T_embed)
        # h1 = self.leakyrelu(h0)
        # h2 = self.linear2(h1)
        # h3 = self.leakyrelu(self.linear3(h2))
        # B = h3
        B = self.reshape_B(B)
        #B = self.mask_B(B, True)
        # for i in range(B.shape[0]):
        #     B[i] = top_k_abs_tensor(B[i], 6)
        B.data = self._preprocess(B.data)
        
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

    def _compute_likelihood(self, X, B):
        X = X.unsqueeze(2)
        if self.equal_variances:
            return 0.5 * self.d * torch.log(
                torch.square(
                    torch.linalg.norm(X - B @ X)
                )
            ) - torch.linalg.slogdet(check_tensor(torch.eye(self.d)) - B)[1]
        else:
            return 0.5 * torch.sum(
                torch.log(
                    torch.sum(
                        torch.square(X - B @ X), dim=0
                    )
                )
            ) - torch.linalg.slogdet(check_tensor(torch.eye(self.d)) - B)[1]

    def _compute_L1_penalty(self, B):
        return torch.norm(B, p=1, dim=(-2, -1)) #+ torch.norm(B, p=1, dim=1)
   
    def _compute_L1_group_penalty(self, B):
        return torch.norm(B, p=2, dim=(0))

    def _compute_h(self, B):
        matrix_exp = torch.exp(B * B)
        traces = torch.sum(torch.diagonal(matrix_exp, dim1=-2, dim2=-1), dim=-1) - self.d
        return traces

    def _compute_smooth_penalty(self,B_t):
        B = B_t.clone().data
        for i in range(self.batch_size):
            b_fft = torch.fft.fft2(B[i])
            b_fftshift = torch.fft.fftshift(b_fft)
            center_idx = b_fftshift.shape[0] // 2
            b_fftshift[center_idx, center_idx] = 0.0
            b_ifft = torch.fft.ifft2(torch.fft.ifftshift(b_fftshift))
            B[i] = b_ifft
            
        return torch.norm(B, p=1, dim=(-2, -1))
    
    def _compute_gradient_penalty(self, loss):
        gradients = torch.autograd.grad(outputs=loss, inputs=self.linear1.parameters(), retain_graph=True)
        gradient_norm1 = torch.sqrt(sum((grad**2).sum() for grad in gradients))
        
        gradients = torch.autograd.grad(outputs=loss, inputs=self.linear1.parameters(), retain_graph=True)
        gradient_norm2 = torch.sqrt(sum((grad**2).sum() for grad in gradients))
        
        return gradient_norm1 + gradient_norm2
    
    def compute_loss(self, X, B, T):
        losses = {}
        total_loss = 0
        X = X - X.mean(axis=0, keepdim=True)
        likelihood = torch.sum(self._compute_likelihood(X, B)) / self.batch_size
        
        for l in self.loss.keys():
            if l == 'L1':
                losses[l] = self.loss[l] * (torch.sum(self._compute_L1_penalty(B)) + torch.sum(self._compute_L1_group_penalty(B))) / self.batch_size
                total_loss += losses[l]
            elif l == 'dag':
                losses[l] = self.loss[l] * torch.sum(self._compute_h(B)) / self.batch_size
                total_loss += losses[l]
            elif l == 'grad':
                losses[l] = self.loss[l] * torch.sum(self._compute_gradient_penalty(B, T)) / self.batch_size
                total_loss += losses[l]
            elif l == 'flat':
                losses[l] = self.loss[l] * torch.sum(torch.pow(B[:, 1:] - B[:, :-1], 2)) / self.batch_size
                total_loss += losses[l]
        
        losses['likelihood'] = likelihood
        losses['total_loss'] = total_loss + likelihood
        #self.gradient.append(self._compute_gradient_penalty(losses['total_loss']).cpu().detach().item())

        return losses
    
    def forward(self, X, T, B_init=None, init_f=False, B_label = None):
        # B = self.generate_B(T)
        #B = self.mlp(T.unsqueeze(1))
        B, T_embed = self.generate_B(T)
        if init_f:
            if B_label is not None:
                label = B_label.reshape(B_label.shape[0], self.d, self.d)
            else:
                label = check_tensor(B_init).repeat(B.shape[0], 1, 1)
            
        self.Bs.append(B.detach().cpu().numpy())
        losses = self.compute_loss(X, B, T_embed) 
        
        if init_f:
            return {'total_loss': torch.nn.functional.mse_loss(B, label)}, B 
        
        return losses
    
    def reshape_B(self, B):
        B_zeros = check_tensor(torch.zeros(B.shape[0], self.d, self.d))
        idx = 0
        for i in range(self.d ** 2):
            row = i // self.d
            col = i % self.d
            if self.d - self.distance <= self.d - row + col <= self.d + self.distance and row != col:
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