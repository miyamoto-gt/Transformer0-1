import torch


class LayerNorm(torch.nn.Module):
    """各トークンの特徴次元(d)だけで正規化する。バッチや系列位置には依存しない。"""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.beta = torch.nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)                 # (B,T,1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)   # (B,T,1)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
