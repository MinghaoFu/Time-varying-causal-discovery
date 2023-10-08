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
    def __init__(self):
        super(TApproximator, self).__init__()
        self.fc1 = nn.Linear(1, 64, bias=False)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.snake = Snake(64, 10)
        self.sigmoid = nn.Sigmoid()
        
        # init.uniform_(self.fc1.bias, -5000, 5000)
        # init.uniform_(self.fc2.bias, -50, 50)
        # init.uniform_(self.fc3.bias, -50, 50)
    def forward(self, x):
        x = self.fc1(x)
        k = check_tensor(torch.rand(x.shape))
        b = k * 4 - 2 - x
        x += b
        x = self.sigmoid(x)
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()

        # Create a long tensor holding the positions
        position = torch.arange(max_len).unsqueeze(1).float()

        # Create a long tensor with the dimensions
        div_term = torch.exp((torch.arange(0, d_model, 2).float()
                              * -(math.log(10000.0) / d_model)))

        # Apply the positional encoding
        pe = torch.empty(max_len, d_model)
        #import ipdb; ipdb.set_trace()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 0::2] = torch.cos(position * div_term)

        # pe = pe.unsqueeze(0)  # Extra dimension for batch size
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('x.shape',x.shape)
        # print(self.pe.shape)
        # import ipdb; ipdb.set_trace()
        x = self.pe[x, :]#.squeeze()
        # x =  self.pe[:x.size(0), :x.size(1)]#.squeeze()
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
        
        self.TApproximators = nn.ModuleList([TApproximator().cuda() for _ in range(self.d ** 2)])
        
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
        T_embed = (1 / self.alpha) * torch.cos(self.beta * T + self.bias)
        #T_embed = (T - T.mean())
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
            ) - torch.linalg.slogdet(torch.eye(self.d).to(self.device) - B)[1]
        else:
            return 0.5 * torch.sum(
                torch.log(
                    torch.sum(
                        torch.square(X - B @ X), dim=0
                    )
                )
            ) - torch.linalg.slogdet(torch.eye(self.d) - B)[1]

    def _compute_L1_penalty(self, B):
        return torch.norm(B, p=1, dim=(-2, -1)) #+ torch.norm(B, p=1, dim=1)
   
    def _compute_L1_group_penalty(self, B):
        return torch.norm(B, p=2, dim=(0))

    def _compute_h(self, B):
        matrix_exp = torch.exp(B * B)
        traces = torch.einsum('bii->b', matrix_exp) - self.d
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
        B.data = self._preprocess(B.data)
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
        #import pdb; pdb.set_trace()
        self.gradient.append(self._compute_gradient_penalty(losses['total_loss']).cpu().detach().item())

        return losses

    # def forward(self, X, T):
    #     B = self.generate_B(T).view(-1, self.d, self.d)
    #     return self.compute_loss(X, B, T)
    
    def forward(self, X, T, B_init=None, init_f=False, B_label = None):
        # B = self.generate_B(T)
        #B = self.mlp(T.unsqueeze(1))
        B, T_embed = self.generate_B(T)
        if init_f:
            if B_label is not None:
                label = B_label.reshape(B_label.shape[0], self.d, self.d)
            else:
                label = torch.from_numpy(B_init).type(torch.FloatTensor).to(B.device).reshape(1, self.d, self.d).repeat(B.shape[0], 1, 1)
            
            
            B = B.view(-1, self.d, self.d)
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
            
            return {'total_loss': torch.nn.functional.mse_loss(B, label)}, B #+ L1_penalty

        B = B.view(-1, self.d, self.d)
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
        self.Bs.append(B.detach().cpu().numpy())
        losses = self.compute_loss(X, B, T_embed) 

        return losses
    

    def step(self, X):
        self.optimizer.zero_grad()
        loss = self.forward(X)
        loss.backward()
        self.optimizer.step()
        return loss.item()