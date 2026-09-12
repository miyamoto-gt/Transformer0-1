import torch
from ffn import FeedForward

torch.manual_seed(0)
d = 32
x = torch.randn(1, 4, d)
ff = FeedForward(d)

# 残差なし: 元の x はどれくらい残るか
h_no = x.clone()
h_re = x.clone()

import torch.nn.functional as F
for i in range(6):
    h_no = ff(h_no)
    h_re = h_re + ff(h_re)
    cos_no = F.cosine_similarity(x[0,0], h_no[0,0], dim=0).item()
    cos_re = F.cosine_similarity(x[0,0], h_re[0,0], dim=0).item()
    print(f"depth {i+1}: 元xとの類似度  なし={cos_no:+.3f}  あり={cos_re:+.3f}")

print("\n--- 勾配 ---")
for use_res in [False, True]:
    x2 = torch.randn(1, 4, d, requires_grad=True)
    h = x2
    for i in range(8):
        h = h + ff(h) if use_res else ff(h)
    h.sum().backward()
    print(f"residual={use_res}: 入力に届いた勾配のノルム = {x2.grad.norm():.6e}")