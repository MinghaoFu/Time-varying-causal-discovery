import torch
import torch.nn as nn

from utils import check_tensor

class golem_loss(nn.Module):
    def __init__(self, args):
        super(golem_loss, self).__init__()
        self.args = args
        
    def forward(self, X, T, B, B_label=None):
        if B_label is not None:
            if self.args.sparse:
                total_loss = ((B - B_label) ** 2).coalesce().values().sum()
            else:
                total_loss = torch.nn.functional.mse_loss(B, B_label)
            losses = {'total_loss': total_loss}
            return losses
        else:
            batch_size = X.shape[0]
            losses = {}
            total_loss = 0
            X = X - X.mean(axis=0, keepdim=True)
            likelihood = torch.sum(self._compute_likelihood(X, B)) / batch_size
            
            for l in self.args.loss.keys():
                if l == 'L1':
                    #  + torch.sum(self._compute_L1_group_penalty(B))
                    losses[l] = self.args.loss[l] * (torch.sum(self._compute_L1_penalty(B))) / batch_size
                    total_loss += losses[l]
                elif l == 'dag':
                    losses[l] = self.args.loss[l] * torch.sum(self._compute_h(B)) / batch_size
                    total_loss += losses[l]
                elif l == 'grad':
                    losses[l] = self.args.loss[l] * torch.sum(self._compute_gradient_penalty(B, T)) / batch_size
                    total_loss += losses[l]
                elif l == 'flat':
                    losses[l] = self.args.loss[l] * torch.sum(torch.pow(B[:, 1:] - B[:, :-1], 2)) / batch_size
                    total_loss += losses[l]
            
            losses['likelihood'] = likelihood
            losses['total_loss'] = total_loss + likelihood
            #self.gradient.append(self._compute_gradient_penalty(losses['total_loss']).cpu().detach().item())

            return losses
        
    def sparse_matmul(self, B, X):
        import pdb; pdb.set_trace()
        inds = check_tensor(self.args.indices).unsqueeze(0).expand(B.shape[0], -1, -1)
        BX = X.new(X.shape)
        BX = torch.gather(X, dim=2, index=inds)
        
        return BX
        
    def _compute_likelihood(self, X, B):
        
        if self.args.sparse:
            import pdb; pdb.set_trace()
            BX = torch.sparse.mm(B, X)
            nnz = self.args.indices.size
            v = check_tensor(torch.tensor([1] * nnz, dtype=torch.int64))
            i = check_tensor(torch.tensor([[i for i in range(nnz)], [i for i in range(nnz)]]))
            I = torch.sparse_coo_tensor(i, v, BX.shape)
        else:
            X = X.unsqueeze(2)
            if self.args.equal_variances:
                return 0.5 * self.d * torch.log(
                    torch.square(
                        torch.linalg.norm(X - B @ X)
                    )
                ) - torch.linalg.slogdet(check_tensor(torch.eye(self.args.d)) - B)[1]
            else:
                return 0.5 * torch.sum(
                    torch.log(
                        torch.sum(
                            torch.square(X - B @ X), dim=0
                        )
                    )
                ) - torch.linalg.slogdet(check_tensor(torch.eye(self.args.d)) - B)[1]

    def _compute_L1_penalty(self, B):
        return torch.norm(B, p=1, dim=(-2, -1)) 
   
    def _compute_L1_group_penalty(self, B):
        return torch.norm(B, p=2, dim=(0))

    def _compute_h(self, B):
        matrix_exp = torch.exp(torch.abs(torch.matmul(B, B)))
        traces = torch.sum(torch.diagonal(matrix_exp, dim1=-2, dim2=-1), dim=-1) - B.shape[1]
        return traces

    def _compute_smooth_penalty(self,B_t):
        B = B_t.clone().data
        batch_size = B.shape[0]
        for i in range(batch_size):
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