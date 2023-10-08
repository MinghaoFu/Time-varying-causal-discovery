import torch
import torch.nn as nn

from utils import check_tensor

class latent_variable_graphical_lasso_loss(nn.Module):
    '''
        latent variable graphical lasso loss
    '''
    def __init__(self, args):
        super(latent_variable_graphical_lasso_loss, self).__init__()
        self.args = args
        
    def forward(self, S, theta, L, alpha, tau):
        batch_size, d, _ = S.shape
        Stheta = torch.sum(torch.diagonal(torch.bmm(S, theta), dim1=-2, dim2=-1), dim=-1)
        g_log_llh = torch.sum(Stheta - torch.linalg.slogdet(theta)[1]) / batch_size
        theta = check_tensor(torch.ones(theta.shape) - torch.eye(d).unsqueeze(0).repeat(batch_size, 1, 1)) * theta
        sparse_term = alpha * torch.linalg.norm(theta) / batch_size
        low_rank_term = tau * torch.linalg.norm(L) / batch_size
        losses = {
            'likelihood': g_log_llh,
            'sparse': sparse_term,
            'low_rank': low_rank_term,
        }
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        return losses
        
    def log_likelihood(self, S, theta, L, alpha, tau):
        return torch.sum(-torch.linalg.slogdet(S, theta - L)[1] + alpha * torch.linalg.norm(theta, 'fro') + tau * torch.linalg.norm(L, 'fro'))
    