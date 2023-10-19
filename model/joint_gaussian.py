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
        I = check_tensor(torch.eye(args.d_X))
        inv_I_minus_B = torch.inverse(I - B)
        est_X_cov = torch.mm(torch.mm(inv_I_minus_B, torch.diag(self.EX_cov)), inv_I_minus_B.t())
        # print(torch.norm(est_X_cov - X_cov).item())
        #print('--- distance: {}'.format(torch.norm(est_X_cov - X_cov).item()))
        
        #
        # L_term_cov = self.C @ self.EL_cov @ self.C.T + self.EX_cov# + 1e-6 * check_tensor(torch.eye(args.d_X))
        # est_X_cov = torch.linalg.inv(I - self.B) @ torch.linalg.inv(L_term_cov) @ torch.linalg.inv(I - self.B).T

        likelihood = model.log_gaussian_likelihood(self.B, torch.diag(self.EX_cov), args.num, X_cov)
        # likelihood = torch.norm(est_X_cov - X_cov, p=1)

        sparsity =  args.sparsity * (torch.norm(B, p=1) + torch.nonzero(B).size(0))
        loss = {}
        loss['likelihood'] = likelihood
        loss['sparsity'] = sparsity
        loss['score'] = sparsity + likelihood

        return loss
    
    @staticmethod
    def log_gaussian_likelihood(B, EX_cov, n, X_cov):
        d_X = X_cov.shape[0]

        I = check_tensor(torch.eye(d_X))
        inv_I_minus_B = torch.inverse(I - B)
        est_X_cov = torch.mm(torch.mm(inv_I_minus_B, EX_cov), inv_I_minus_B.t())
        
        return 0.5 * (torch.slogdet(est_X_cov)[1] + \
            torch.trace(torch.mm(torch.inverse(est_X_cov), X_cov)) + \
                d_X * torch.log(check_tensor(torch.tensor(torch.pi * 2))))