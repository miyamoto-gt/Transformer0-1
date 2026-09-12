import torch
from model import GPTMini
from data_utils import load_data, get_batch
import time

torch.manual_seed(1337)

block_size, batch_size = 128, 32
d_model, n_layer = 64, 4
max_iters, eval_interval, eval_iters = 3000, 300, 100
lr = 1e-3

train, val, stoi, itos = load_data()
V = len(stoi)
m = GPTMini(V, d_model, block_size, n_layer)
opt = torch.optim.AdamW(m.parameters(), lr=lr)
print("params:", sum(p.numel() for p in m.parameters()))


@torch.no_grad()
def estimate_loss():
    m.eval()
    out = {}
    for name, data in [("train", train), ("val", val)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size)
            _, loss = m(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    m.train()
    return out

t0 = time.time()
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        l = estimate_loss()
        print(f"iter {it:5d}  train {l['train']:.4f}  validation {l['val']:.4f}")
    x, y = get_batch(train, block_size, batch_size)
    _, loss = m(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
elapsed = time.time() - t0
print(f"elapsed: {elapsed:.2f} sec")
torch.save(m.state_dict(), f"ckpt_bs{block_size}.pt")
print("saved ckpt.pt")
