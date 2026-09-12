"""n_head / block_size / d_model を1つずつ変えて学習し、最終lossを比べる。

train.py と同じ設定(seed=1337, lr=1e-3, batch=32, n_layer=4)で、
GPTMini が Block に渡す n_head だけ差し替えている。

  python check_sweep.py                      # 既定値(train.py と同じ)で実行
  python check_sweep.py --n-head 1           # ヘッド数だけ変える
  python check_sweep.py --block-size 32      # 文脈長だけ変える
  python check_sweep.py --d-model 128        # 幅だけ変える
  python check_sweep.py --iters 300          # 短く試す
"""
import argparse, time, torch
import block as B, model as M
from data_utils import load_data, get_batch

ap = argparse.ArgumentParser(
    description="n_head / block_size / d_model を振って最終lossを比べる",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
ap.add_argument("--n-head",     type=int, default=4,    help="ヘッド数 (d_model を割り切ること)")
ap.add_argument("--block-size", type=int, default=128,  help="文脈長")
ap.add_argument("--d-model",    type=int, default=64,   help="モデルの幅")
ap.add_argument("--iters",      type=int, default=3000, help="学習ステップ数")
args = ap.parse_args()

NH, BS, DM, ITERS = args.n_head, args.block_size, args.d_model, args.iters
if DM % NH != 0:
    ap.error(f"d_model={DM} は n_head={NH} で割り切れません")

_Block = B.Block
class Blk(_Block):                       # GPTMini は n_head を渡さないので既定値を差し替える
    def __init__(self, d, bs, n_head=NH, pre_ln=True):
        super().__init__(d, bs, NH, pre_ln)
M.Block = Blk

torch.manual_seed(1337)
batch_size, eval_iters, lr = 32, 100, 1e-3
train, val, stoi, itos = load_data()
m = M.GPTMini(len(stoi), DM, BS, 4)
opt = torch.optim.AdamW(m.parameters(), lr=lr)
tag = f"n_head={NH} block_size={BS} d_model={DM}"
print(f"### {tag}  params={sum(p.numel() for p in m.parameters())}", flush=True)


@torch.no_grad()
def estimate_loss():
    m.eval(); out = {}
    for name, data in [("train", train), ("val", val)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, BS, batch_size)
            _, loss = m(x, y); losses[k] = loss.item()
        out[name] = losses.mean().item()
    m.train(); return out


t0 = time.time()
for it in range(ITERS + 1):
    if it % 300 == 0:
        l = estimate_loss()
        print(f"iter {it:5d}  train {l['train']:.4f}  validation {l['val']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    x, y = get_batch(train, BS, batch_size)
    _, loss = m(x, y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print(f"### DONE {tag}  elapsed {time.time()-t0:.1f}s", flush=True)
