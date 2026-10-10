# ce mse mse_lr*10の学習ログをまとめてグラフ化する
import os, glob, csv
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(current_dir, "temp")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for path in sorted(glob.glob(os.path.join(out_dir, "log_*.csv"))):
    tag = os.path.basename(path)[4:-4]          # log_a_ce.csv → a_ce
    with open(path) as f:
        rows = list(csv.DictReader(f))
    it = [int(r["iter"]) for r in rows]
    axes[0].plot(it, [float(r["val_ce"]) for r in rows], marker="o", label=tag)
    axes[1].plot(it, [float(r["val_acc"]) for r in rows], marker="o", label=tag)
    axes[2].plot(it, [float(r["lm_head_grad"]) for r in rows], marker="o", label=tag)

axes[0].set_title("val CE (lower is better)")
axes[1].set_title("val top-1 accuracy")
axes[2].set_title("lm_head grad norm (log scale)")
axes[2].set_yscale("log")                       # A と B の桁の違いを見るため対数軸
for ax in axes:
    ax.set_xlabel("iter")
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
out = os.path.join(out_dir, "compare.png")
plt.savefig(out, dpi=150)
print("saved:", out)