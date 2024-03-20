import torch
from torch import nn
import torch.nn.functional as F

import math
from inspect import isfunction

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

class Experts(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        ):
        super().__init__()

        hidden_dim = default(hidden_dim, dim)
        
        w1 = torch.zeros(num_experts, dim, hidden_dim)
        w2 = torch.zeros(num_experts, hidden_dim, dim//4)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = nn.GELU()

    def forward(self, x):
        hidden = torch.einsum('...nd,edh->...enh', x, self.w1)
        hidden = self.act(hidden)
        out    = torch.einsum('...enh,...ehd->...nde', hidden, self.w2)
        return out

class Gating(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        num_tasks = 2002,
        ):
        super().__init__()

        g = torch.zeros(dim, num_tasks, num_experts)
        g = init_(g)
        self.g = nn.Parameter(g)
        self.act = nn.Softmax(dim=-1)

    def forward(self, x):
        gating = torch.einsum('...d,...dte->...te', x, self.g)
        gating = self.act(gating)
        return gating

class Tower(nn.Module):
    def __init__(self,
        dim,
        num_tasks = 2002,
        hidden_dim = 8,
        dropout = 0.1,
        ):
        super().__init__()

        hidden_dim = default(hidden_dim, dim)
        
        w1 = torch.zeros(num_tasks, dim, hidden_dim)
        w2 = torch.zeros(num_tasks, hidden_dim, 1)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = torch.einsum('...td,...tdh->...th', x, self.w1)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        out    = torch.einsum('...th,...thd->...td', hidden, self.w2)
        return out

class MMOE(nn.Module):
    def __init__(
        self,
        dim,
        num_experts = 16,
        num_tasks = 2002, 
        dropout = 0.1
        ):
        super().__init__()
        self.experts = Experts(dim, num_experts)
        self.gates = Gating(dim, num_experts, num_tasks)
        self.tower = Tower(dim//4, dropout = dropout, num_tasks=num_tasks)
    
    def forward(self, x):
        g = self.gates(x)
        e = self.experts(x)
        out = torch.einsum('...de,...te->...td', e, g)
        out = self.tower(out)
        out = out.squeeze(-1)
        return out