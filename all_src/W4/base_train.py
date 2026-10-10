"""W4 基準ファイル: 学習・評価・保存の共通部分
実験ファイルは loss_fn だけを定義して run() を呼ぶ。
"""
import os, sys, csv

current_dir = os.path.dirname(os.path.abspath(__file__))
W2_DIR = os.path.abspath(os.path.join(current_dir, "..", "W2"))
sys.path.insert(0, W2_DIR)

import torch
import torch.nn.functional as F
from model import GPTMini
from data_utils import load_data

# ---- 固定する設定(W2 と同じ値にそろえる) --------------------------------
lr = 1e-3           # ← W2 の train.py の学習率に合わせる
block_size, batch_size =32, 32
d_model, n_layer = 64, 4             # GPTMini の4つ目の引数
max_iters, eval_interval, eval_iters = 3000, 300, 50
seed = 0

out_dir = os.path.join(current_dir, "temp")
os.makedirs(out_dir, exist_ok=True)

train, val, stoi, itos = load_data(os.path.join(W2_DIR, "data", "input.txt"))
V = len(stoi)


def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(m):
    """共通の物差し: どの実験でも val CE と正解率で測る"""
    m.eval()
    ces, accs = [], []
    for _ in range(eval_iters):
        x, y = get_batch(val)
        logits, _ = m(x)                                   # targets なし → (B, T, V)
        ces.append(F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).item())
        accs.append((logits.argmax(-1) == y).float().mean().item())
    m.train()
    return sum(ces) / len(ces), sum(accs) / len(accs)


def run(loss_fn, tag, lr=lr):
    torch.manual_seed(seed)
    m = GPTMini(V, d_model, block_size, n_layer)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)# adamW で学習するのが W2 の train.py と同じ

    rows = []
    print(f"[{tag}] {'iter':>5} | {'学習loss':>8} | {'val CE':>7} | {'val acc':>7} | {'lm_head勾配':>10}")
    for it in range(max_iters + 1):
        x, y = get_batch(train)
        logits, _ = m(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()

        if it % eval_interval== 0:
            gnorm = m.lm_head.weight.grad.norm().item()  # ← 出力層の名前が違えば直す
            vce, vacc = evaluate(m)
            rows.append([it, loss.item(), vce, vacc, gnorm])
            print(f"[{tag}] {it:>5} | {loss.item():>8.4f} | {vce:>7.4f} | {vacc:>7.3f} | {gnorm:>10.2e}")

        opt.step()

    with open(os.path.join(out_dir, f"log_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "train_loss", "val_ce", "val_acc", "lm_head_grad"])
        w.writerows(rows)
    torch.save(m.state_dict(), os.path.join(out_dir, f"ckpt_{tag}.pt"))

    # 生成サンプル(T=1.0)
    m.eval()
    torch.manual_seed(0)
    idx = torch.zeros((1, 1), dtype=torch.long)
    with torch.no_grad():
        for _ in range(500):
            logits, _ = m(idx[:, -block_size:])
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
    with open(os.path.join(out_dir, f"sample_{tag}.txt"), "w") as f:
        f.write("".join(itos[i.item()] for i in idx[0]))