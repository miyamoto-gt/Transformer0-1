# W5: log_{tag}.csv を読んで train/val CE・accuracy・gap を描く
import os, glob, csv, math
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(current_dir, "temp")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for path in sorted(glob.glob(os.path.join(out_dir, "log_*.csv"))):
    tag = os.path.basename(path)[4:-4]          # log_base.csv → base
    with open(path) as f:
        rows = list(csv.DictReader(f))
    it = [int(r["iter"]) for r in rows]
    tr = [float(r["train_ce"]) for r in rows]
    va = [float(r["val_ce"]) for r in rows]
    gap = [v - t for v, t in zip(va, tr)]       

    axes[0].plot(it, va, marker="o", label=tag + "_val")
    axes[0].plot(it, tr, linestyle="--", label=tag + "_train")
    axes[1].plot(it, [float(r["val_acc"]) for r in rows], marker="o", label=tag + "_val")
    axes[1].plot(it, [float(r["train_acc"]) for r in rows], linestyle="--", label=tag + "_train")
    axes[2].plot(it, gap, marker="o", label=tag + "_gap")

axes[0].axhline(y=math.log(65), color="gray", linestyle=":", label="ln(65)")
axes[2].axhline(y=0, color="gray", linestyle=":")

axes[0].set_title("CE: train / val")
axes[1].set_title("accuracy: train / val")
axes[2].set_title("gap = val CE − train CE")
for ax in axes:
    ax.set_xlabel("iter")
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
out = os.path.join(out_dir, "curve_" + tag + ".png")
plt.savefig(out, dpi=150)
print("saved:", out)