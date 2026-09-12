import torch
from block import Block

d, T, block_size, n_layer = 32, 6, 8, 12

for pre in [True, False]:
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList([Block(d, block_size, pre_ln=pre) for _ in range(n_layer)])
    x = torch.randn(1, T, d, requires_grad=True)

    h = x
    stds = []
    for blk in blocks:
        h = blk(h)
        stds.append(h.std().item())

    h.sum().backward()
    name = "Pre-LN " if pre else "Post-LN"
    print(f"{name}: std {stds[0]:.3f} -> {stds[-1]:.3f}   input grad = {x.grad.norm():.4e}")