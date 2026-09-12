import torch
from attention import MultiHeadAttention
from ffn import FeedForward
from layernorm import LayerNorm

class Block(torch.nn.Module):
    def __init__(self, d_model, block_size, n_head=4, pre_ln=True):
        super().__init__()
        self.pre_ln = pre_ln
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, block_size)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model)
    def forward(self, x):
        if self.pre_ln:
            a, _, _ = self.attn(self.ln1(x))
            x = x + a
            x = x + self.ffn(self.ln2(x))
        else:
            a, _, _ = self.attn(x)
            x = self.ln1(x + a)          # 足してから正規化
            x = self.ln2(x + self.ffn(x))
        return x