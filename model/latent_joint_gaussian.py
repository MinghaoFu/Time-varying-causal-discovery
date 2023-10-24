import torch
import torch.nn as nn
from utils import check_tensor

class model(nn.Module):
    def __init__(self, args, B_init=None, C_init=None, EL_cov=None, EX_cov=None) -> None:
        super(model, self).__init__()
        self.B = torch.nn.Parameter(check_tensor(torch.randn(args.d_X, args.d_X, requires_grad=True))) if B_init == None else torch.nn.Parameter(check_tensor(B_init.clone().detach().requires_grad_(True)))
        with torch.no_grad():
            self.B.fill_diagonal_(0)
        self.C = torch.nn.Parameter(check_tensor(torch.randn(args.d_X, args.d_L, requires_grad=True))) if C_init == None else torch.nn.Parameter(check_tensor(C_init.clone().detach().requires_grad_(True)))
        
        self.EL_cov = torch.nn.Parameter(check_tensor(torch.randn(args.d_L, requires_grad=True)))
        self.EX_cov = torch.nn.Parameter(check_tensor(torch.randn(args.d_X, requires_grad=True))) # ones
        
        self.n_params = sum(p.view(-1).size(0) for p in self.parameters())
        
    def forward(self, args, X_cov):
        B = self.B
        #B[B < args.graph_thres].fill_(0)
        #B.masked_fill_(torch.abs(B_i) < self.tol, 0)

        self.est_X_cov, likelihood = model.log_gaussian_likelihood(B, self.C, torch.diag(self.EX_cov), torch.diag(self.EL_cov), args.num, X_cov)
        sparsity = args.sparsity * torch.norm(B, p=1)# + torch.sum(torch.abs(B) >= args.graph_thres)
        dag = args.DAG * torch.sum(torch.diagonal(torch.matrix_exp(B * B), dim1=-2, dim2=-1)) - args.d_X
        loss = {}
        loss['likelihood'] = likelihood
        loss['sparsity'] = sparsity
        loss['dag'] = dag
        loss['score'] = (torch.sum(torch.abs(B) >= args.graph_thres) + 2) * torch.log(check_tensor(torch.tensor(args.num))) + 2 * likelihood + sparsity + dag
        
        
        return loss
    
    @staticmethod
    def log_gaussian_likelihood(B, C, EX_cov, EL_cov, n, X_cov):
        d_X = X_cov.shape[0]

        I = check_tensor(torch.eye(d_X))
        inv_I_minus_B = torch.inverse(I - B)
        est_X_cov = inv_I_minus_B.T @ (C @ EL_cov @ C.t() + EX_cov) @ inv_I_minus_B
        if torch.det(est_X_cov) < -1:
            print(torch.det(est_X_cov))
        return est_X_cov, 0.5 * (torch.slogdet(est_X_cov)[1] + \
            torch.trace(torch.mm(torch.inverse(est_X_cov), X_cov)) + \
                d_X * torch.log(check_tensor(torch.tensor(torch.pi * 2))))