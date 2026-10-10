"""保存済みの重みを、複数の物差しで採点する
使い方: uv run python eval_ckpt.py ce mse
  → temp/ckpt_a_ce.pt と temp/ckpt_mse.pt を読む

val 全体を block_size ごとに区切って全部評価するので、ランダム性がなく毎回同じ値になる。
"""
import os, sys
import torch
import torch.nn.functional as F
from base_train import GPTMini, V, val, d_model, n_layer, out_dir

tags = sys.argv[1:] or ["ce", "mse", "mse_lr*10"]  # コマンドライン引数がなければすべて評価する


@torch.no_grad()
def score(m, block):
    n = (len(val) - 1) // block
    x = val[: n * block].view(n, block)
    y = val[1 : n * block + 1].view(n, block)
    pc_all, mse_all, acc_all = [], [], []
    for i in range(0, n, 256):
        xb, yb = x[i : i + 256], y[i : i + 256]
        logits, _ = m(xb)
        probs = torch.softmax(logits, dim=-1)
        pc_all.append(probs.gather(-1, yb.unsqueeze(-1)).squeeze(-1).flatten())   # 正解の文字に置いた確率
        mse_all.append(((probs - F.one_hot(yb, V).float()) ** 2).sum(-1).flatten())
        acc_all.append((logits.argmax(-1) == yb).float().flatten())
    pc = torch.cat(pc_all)
    return {
        "val CE": (-torch.log(pc)).mean().item(),
        "val MSE": torch.cat(mse_all).mean().item(),
        "正解率(%)": torch.cat(acc_all).mean().item()*100,
        "p中央値": pc.median().item(),
        "p<0.01(%)": (pc < 0.01).float().mean().item()*100,
        "p<0.001(%)": (pc < 0.001).float().mean().item()*100,
    }


results = {}
for tag in tags:
    sd = torch.load(os.path.join(out_dir, f"ckpt_{tag}.pt"))
    block = sd["pos_emb.weight"].shape[0]
    m = GPTMini(V, d_model, block, n_layer)
    m.load_state_dict(sd)
    m.eval()
    results[tag] = score(m, block)

keys = list(next(iter(results.values())).keys())
print(f"{'':>8} | " + " | ".join(f"{k:>8}" for k in keys))
for tag, r in results.items():
    print(f"{tag:>8} | " + " | ".join(f"{r[k]:>8.4f}" for k in keys))