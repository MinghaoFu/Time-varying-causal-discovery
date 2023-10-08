import torch
import torch.nn as nn

from .base import check_tensor

def top_k_abs_tensor(tensor, k):
    d = tensor.shape[0]
    abs_tensor = torch.abs(tensor)
    _, indices = torch.topk(abs_tensor.view(-1), k)
    
    flat_tensor = tensor.view(-1)
    flat_zero_tensor = torch.zeros_like(flat_tensor)
    flat_zero_tensor[indices] = flat_tensor[indices]
    
    zero_tensor = check_tensor(flat_zero_tensor.view(d, d))
    
    
    # batch_size, d, _ = tensor.shape
    # values, indices = torch.topk(tensor.view(batch_size, -1), k=k, dim=-1)
    # result = torch.zeros_like(tensor).view(batch_size, -1)
    # result.scatter_(1, indices, values)
    # result = result.view(batch_size, d, d)
    return zero_tensor