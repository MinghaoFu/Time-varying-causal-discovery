import torch
import torch.nn as nn
from utils import check_tensor

class model(nn.Module):
    def __init__(self, args, B_init=None, C_init=None, EL_cov=None, EX_cov=None) -> None:
        super(model, self).__init__()
        self.B = torch.nn.Parameter(check_tensor(torch.rand((args.d_X, args.d_X), requires_grad=True))) if B_init == None else torch.nn.Parameter(check_tensor(B_init.clone().detach().requires_grad_(True)))
        with torch.no_grad():
            self.B.fill_diagonal_(0)

        self.EX_cov = torch.nn.Parameter(check_tensor(torch.randn(args.d_X, requires_grad=True)))
        
    def forward(self, args, X_cov):
        B = self.B
        self.est_X_cov, likelihood = model.log_gaussian_likelihood(B, torch.diag(self.EX_cov), args.num, X_cov)
        # likelihood = torch.norm(est_X_cov - X_cov, p=1)
        sparsity = torch.norm(B, p=1) #+ torch.nonzero(B).size(0)
        mat_exp = torch.matrix_exp(B * B)
        dag = torch.sum(torch.diagonal(mat_exp, dim1=-2, dim2=-1)) - args.d_X
        loss = {}
        loss['likelihood'] = likelihood
        loss['sparsity'] = sparsity
        loss['dag'] = dag
        loss['score'] = likelihood + args.sparsity * sparsity + args.DAG * dag
        
        return loss
    
    @staticmethod
    def log_gaussian_likelihood(B, EX_cov, n, X_cov):
        d_X = X_cov.shape[0]

        I = check_tensor(torch.eye(d_X))
        inv_I_minus_B = torch.inverse(I - B)
        est_X_cov = inv_I_minus_B.T @ EX_cov @ inv_I_minus_B
        if torch.det(est_X_cov) < -1:
            print(torch.det(est_X_cov))
        return est_X_cov, 0.5 * (torch.slogdet(est_X_cov)[1] + \
            torch.trace(torch.mm(torch.inverse(est_X_cov), X_cov)) + \
                d_X * torch.log(check_tensor(torch.tensor(torch.pi * 2))))