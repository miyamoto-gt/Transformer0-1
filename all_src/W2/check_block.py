import torch
from block import Block

torch.manual_seed(0)
B, T, d = 2, 6, 32
x = torch.randn(B, T, d)
blk = Block(d, block_size=128)

print("shape:", blk(x).shape)

total = 0
for n, p in blk.named_parameters():
    print(f"{n:20s} {tuple(p.shape)}")
    total += p.numel()
expect = 3*d*d + (d*d+d) + (4*d*d+4*d) + (4*d*d+d) + 4*d
print(f"total params: {total}  (expect {expect})")

x2 = x.clone()
x2[:, 3:, :] = torch.randn(B, T - 3, d)
with torch.no_grad():
    print("causal:", torch.allclose(blk(x)[:, :3], blk(x2)[:, :3], atol=1e-6))

blocks = [Block(d, block_size=128) for _ in range(8)]
h = x.clone()
with torch.no_grad():
    for i, b in enumerate(blocks):
        h = b(h)
        r = torch.linalg.matrix_rank(h[0], rtol=1e-4).item()
        print(f"depth {i+1}: std={h.std():.3f} rank={r} "
              f"(sqrt予測 {(i+2)**0.5:.3f})")

delta = b(h) - h
print(f"  |delta|/|h| = {delta.std()/h.std():.3f}")