import common, torch
from broken import BlockNoResidual

for name, broken in [("standard", False), ("no_residual", True)]:
    m = common.build_model()
    if broken:
        for b in m.blocks:
            b.__class__ = BlockNoResidual
    x, y = common.get_batch(common.train, common.block_size, common.batch_size)
    _, loss = m(x, y)
    loss.backward()
    print(name, [f"{common.block_grad_norm(b):.4f}" for b in m.blocks])

m=common.build_model()
x, y = common.get_batch(common.train, common.block_size, common.batch_size)
y = y[torch.randperm(y.size(0))]
_, loss = m(x, y)
loss.backward()
print("shuffle", [f"{common.block_grad_norm(b):.4f}" for b in m.blocks])