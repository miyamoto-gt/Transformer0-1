"""W4 実験2: 温度 T ごとの生成比較(学習はしない)
使い方:
  uv run python sample_temp.py                     
  uv run python sample_temp.py --ckpt ckpt_a_ce.pt # 実験1-A の重み
  uv run python sample_temp.py --ckpt ckpt_b_mse.pt # 実験1-B の重み
"""
import os, argparse
import torch
from base_train import GPTMini, V, train, itos, d_model, n_layer, W2_DIR, current_dir,out_dir

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=os.path.join(W2_DIR, "ckpt.pt"))
p.add_argument("--n", type=int, default=1000)
args = p.parse_args()
ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(out_dir, args.ckpt)
tag = "w2" if args.ckpt.endswith(os.path.join("W2", "ckpt.pt")) else os.path.splitext(os.path.basename(ckpt_path))[0]

sd = torch.load(ckpt_path)
block = sd["pos_emb.weight"].shape[0]      # ckpt から block_size を読む
m = GPTMini(V, d_model, block, n_layer)
m.load_state_dict(sd)
m.eval()

# 実在単語の判定用: 訓練データに出てくる単語の集合
train_text = "".join(itos[i] for i in train.tolist())
def words(s):
    return [w.strip(".,;:!?'\"-").lower() for w in s.split() if w.strip(".,;:!?'\"-")]
train_words = set(words(train_text))


@torch.no_grad()
def generate(T, n, seed=0):
    torch.manual_seed(seed)                     
    idx = torch.zeros((1, 1), dtype=torch.long)
    ents = []
    for _ in range(n):
        logits, _ = m(idx[:, -block:])
        logits = logits[:, -1, :]
        probs = torch.softmax(logits / T, dim=-1)   
        ents.append(-(probs * torch.log2(probs + 1e-12)).sum().item())
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, nxt), dim=1)
    text = "".join(itos[i.item()] for i in idx[0])
    return text, sum(ents) / len(ents)


print(f"[{tag}] {'T':>4} | {'エントロピー(bit)':>10} | {'単語の重複率':>8} | {'実在単語率':>8}")
for T in [0.5, 0.8, 1.0, 1.5, 2.0]:
    text, ent = generate(T, args.n)
    ws = words(text)
    dup = 1 - len(set(ws)) / len(ws)
    real = sum(w in train_words for w in ws) / len(ws)
    print(f"[{tag}] {T:>4} | {ent:>10.3f} | {dup:>8.3f} | {real:>8.3f}")
    with open(os.path.join(out_dir, f"sample_{tag}_T{T}.txt"), "w") as f:
        f.write(text)