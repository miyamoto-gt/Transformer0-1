import torch
from model import GPTMini
from data_utils import load_data

torch.manual_seed(0)
train, val, stoi, itos = load_data()
V = len(stoi)

m = GPTMini(V, 64, 32, 4)
m.load_state_dict(torch.load("ckpt.pt"))
m.eval()

idx = torch.zeros((1, 1), dtype=torch.long)   # 改行から開始
for _ in range(500):
    idx_cond = idx[:, -32:]                    # block_size に切る
    logits, _ = m(idx_cond)
    logits = logits[:, -1, :]                  # 最後の位置だけ
    probs = torch.softmax(logits, dim=-1)
    nxt = torch.multinomial(probs, num_samples=1)
    idx = torch.cat((idx, nxt), dim=1)

print("".join(itos[i.item()] for i in idx[0]))