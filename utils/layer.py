import torch
import torch.nn as nn

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0, d_model, 2).float()
                              * -(math.log(10000.0) / d_model)))

        pe = torch.empty(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 0::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[x, :]
        return x