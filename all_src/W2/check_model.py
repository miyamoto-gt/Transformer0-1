import torch
from model import GPTMini
from data_utils import load_data, get_batch

torch.manual_seed(0)
train, val, stoi, itos = load_data()
V = len(stoi)
print("vocab size:", V)
m = GPTMini(vocab_size=V, d_model=64, block_size=32, n_layer=4)
print("params:", sum(p.numel() for p in m.parameters()))

x, y = get_batch(train, block_size=32, batch_size=4)
logits, loss = m(x, y)
print("logits:", logits.shape)
print("loss:", loss.item())
print("expected (random):", torch.log(torch.tensor(float(V))).item())