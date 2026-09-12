import torch.nn as nn
import torch
class Head(nn.Module):
    def __init__(self,d_model,head_size,block_size):
        super().__init__()
        self.key = nn.Linear(d_model,head_size,bias=False)
        self.query = nn.Linear(d_model,head_size,bias=False)
        self.value = nn.Linear(d_model,head_size,bias=False)
        self.register_buffer("mask",torch.tril(torch.ones(block_size,block_size)).bool())

    def forward(self,x):
        B,T,C=x.shape
        q=self.query(x)
        k=self.key(x)
        v=self.value(x)

        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B,T,T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        P = torch.softmax(scores, dim=-1)
        out = P @ v
        return out ,P,scores

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_size = d_model // n_head
        self.qkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = torch.nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())

    def forward(self, x):
        B, T, C = x.shape
        nh, hs = self.n_head, self.head_size

        q, k, v = self.qkv(x).split(C, dim=2)              # each (B,T,C)
        q = q.view(B, T, nh, hs).transpose(1, 2)           # (B,nh,T,hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) * hs ** -0.5      # (B,nh,T,T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        P = torch.softmax(scores, dim=-1)
        out = P @ v                                         # (B,nh,T,hs)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), P, scores