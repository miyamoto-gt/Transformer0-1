"""W5 基準ファイル: 学習・評価・保存の共通部分
実験ファイルは run() の引数を1つだけ変えて呼ぶ(損失はCE固定)。
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
eval_iters=50
seed = 0

out_dir = os.path.join(current_dir, "temp")
os.makedirs(out_dir, exist_ok=True)

train, val, stoi, itos = load_data(os.path.join(W2_DIR, "data", "input.txt"))
V = len(stoi)


def get_batch(data,generator=None):
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

def make_eval_batches(data,seed):
    generator= torch.Generator()
    generator.manual_seed(seed)
    batches=[]
    for _ in range(eval_iters):
        batches.append(get_batch(data,generator=generator))
    return batches

@torch.no_grad()
def evaluate(m,batches):
    """共通の物差し: 渡した評価バッチの CE と正解率を測る"""
    m.eval()
    ces, accs = [], []
    for x,y in batches:
        logits, _ = m(x)                                   # targets なし → (B, T, V)
        ces.append(F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).item())
        accs.append((logits.argmax(-1) == y).float().mean().item())
    m.train()
    return sum(ces) / len(ces), sum(accs) / len(accs)


def run(loss_fn, tag, lr=lr,weight_decay=0,d_model=64,n_layer=4,max_iters=3000,eval_interval=150,train_frac=1.0):
    train_range=int(len(train)*train_frac)
    train_data=train[:train_range]
    print("Train data size:", len(train_data))
    run_dir=os.path.join(out_dir,tag)
    os.makedirs(run_dir,exist_ok=True)
    train_eval_batches=make_eval_batches(train_data, seed)
    val_eval_batches=make_eval_batches(val, seed)
    torch.manual_seed(seed)
    m = GPTMini(V, d_model, block_size, n_layer)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)  # adamW で学習するのが W2 の train.py と同じ
    print("weight_decay:",opt.param_groups[0]['weight_decay'])  # weight_decay の値を確認する
    rows = []
    print(f"[{tag}] {'iter':>5} | {'train CE':>8} | {'train acc':>9} | {'val CE':>7} | {'val acc':>7} | {'lm_head勾配':>10}")
    for it in range(max_iters + 1):
        x, y = get_batch(train_data)
        logits, _ = m(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()

        if it % eval_interval== 0:
            gnorm = m.lm_head.weight.grad.norm().item()  # ← 出力層の名前が違えば直す
            tce,tacc=evaluate(m,train_eval_batches)
            vce,vacc=evaluate(m,val_eval_batches)
            rows.append([it, tce,tacc, vce, vacc, gnorm])
            print(f"[{tag}] {it:>5} | {tce:>8.4f} | {tacc:>7.3f} | {vce:>7.4f} | {vacc:>7.3f} | {gnorm:>10.2e}")

        opt.step()

    with open(os.path.join(run_dir, f"log_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "train_ce", "train_acc", "val_ce", "val_acc", "lm_head_grad"])
        w.writerows(rows)
    torch.save(m.state_dict(), os.path.join(run_dir, f"ckpt_{tag}.pt"))

    # 生成サンプル(T=1.0)
    m.eval()
    torch.manual_seed(0)
    idx = torch.zeros((1, 1), dtype=torch.long)
    with torch.no_grad():
        for _ in range(500):
            logits, _ = m(idx[:, -block_size:])
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
    with open(os.path.join(run_dir, f"sample_{tag}.txt"), "w") as f:
        f.write("".join(itos[i.item()] for i in idx[0]))