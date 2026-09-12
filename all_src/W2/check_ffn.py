import torch
from ffn import FeedForward

torch.manual_seed(0)
B, T, d = 2, 6, 32 # バッチ　２　、系列長　６、次元　３２
x = torch.randn(B, T, d)
ff = FeedForward(d)

# (1) 入口と出口が同じ d か
print("shape:", ff(x).shape)

# (2) weight の shape（論文の W1 と転置になっている）
for name, p in ff.named_parameters():
    print(name, tuple(p.shape))

# (3) 位置独立か
perm = torch.randperm(T)
print("position-wise:", torch.allclose(ff(x)[:, perm], ff(x[:, perm]), atol=1e-6))

# (4) clamp が効いているか
# f(a+b) - (f(a)+f(b)) では bias の分が残るだけで、ReLU を外しても 0 にならない。
# アフィン写像 f(x)=Ax+c なら 2階差分 f(a+b)-f(a)-f(b)+f(0) が厳密に 0 になるので、
# こちらを見ないと「非線形かどうか」は判定できない。
a, b = torch.randn(1, 1, d), torch.randn(1, 1, d)
z = torch.zeros(1, 1, d)
affine = lambda t: ff.w2(ff.w1(t))          # clamp を挟まない＝アフィン写像

print(f"{'':22s}{'f(a+b)-(f(a)+f(b))':>22s}{'2階差分':>16s}")
for name, f in [("ReLUあり(本物)", ff), ("ReLUなし(アフィン)", affine)]:
    g1 = (f(a + b) - (f(a) + f(b))).abs().max().item()
    g2 = (f(a + b) - f(a) - f(b) + f(z)).abs().max().item()
    print(f"{name:22s}{g1:22.6f}{g2:16.6f}")
print("→ 左列はReLUを外しても0にならない（biasの分）ので非線形性の証明にならない")
print("→ 右列はアフィンなら厳密に0。これなら証拠になる")

# (5) パラメータ数
print("params:", sum(p.numel() for p in ff.parameters()), "expected:", 8*d*d + 5*d)