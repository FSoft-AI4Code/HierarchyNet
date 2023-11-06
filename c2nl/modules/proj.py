import torch
import torch.nn as nn

class OneNonLinearProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.seq_module = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace = True),
        )
    def forward(self, x):
        return self.seq_module(x)
class TwoNonLinearProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.seq_module = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace = True),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.seq_module(x)