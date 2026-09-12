import torch
from attention import Head

torch.manual_seed(0)
B,T,d,hs=2,6,32,32
x=torch.randn(B,T,d)
h=Head(d,hs,block_size=8)
out ,P,scores=h(x)
print("out:", out.shape, " P:", P.shape)

# 因果性: 上三角が0か
print("causal:", torch.allclose(P[0].triu(1), torch.zeros(T, T)))

# 各行の和が1か
print("row sum:", P[0].sum(dim=-1))

# 位置独立でない（FFN との対比）
perm = torch.randperm(T)
print("position-wise:", torch.allclose(h(x)[0][:, perm], h(x[:, perm])[0], atol=1e-6))