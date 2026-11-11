# W5: temp/<tag>/log_<tag>.csv を読んで train/val CE・accuracy・gap を描く
# 例: uv run python plot.py base data_0.1_6000 data_0.01
import os, sys, csv, math
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(current_dir, "temp")

tags = sys.argv[1:]
if not tags:
    sys.exit("描く tag を指定してください 例: uv run python plot.py base")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, tag in enumerate(tags):              # 指定した順に色を決める
    path = os.path.join(out_dir, tag, f"log_{tag}.csv")
    if not os.path.exists(path):
        print("見つからない:", path)
        continue
    with open(path) as f:
        rows = list(csv.DictReader(f))
    it = [int(r["iter"]) for r in rows]
    tr = [float(r["train_ce"]) for r in rows]
    va = [float(r["val_ce"]) for r in rows]
    gap = [v - t for v, t in zip(va, tr)]
    # tag ごとに同じ系統の色: val = 濃い色、train = 薄い色(tab20 は濃淡が2色1組)
    cmap = plt.get_cmap("tab20")
    c_val, c_train = cmap(2 * i), cmap(2 * i + 1)

    # CE
    axes[0].plot(it, va, marker="o", markersize=4, color=c_val, label=tag + "_val")
    axes[0].plot(it, tr, linestyle="--", linewidth=2, color=c_train, label=tag + "_train")
    k = va.index(min(va))                    # val の最小値
    axes[0].scatter(it[k], va[k], marker="*", s=220, color=c_val,
                    edgecolor="black", zorder=5)

    # accuracy
    axes[1].plot(it, [float(r["val_acc"]) for r in rows], marker="o", markersize=4, color=c_val)
    axes[1].plot(it, [float(r["train_acc"]) for r in rows], linestyle="--", linewidth=2, color=c_train)

    # gap(val と同じ濃い色)
    axes[2].plot(it, gap, marker="o", markersize=4, color=c_val)

axes[0].axhline(y=math.log(65), color="gray", linestyle=":", label="ln(65)")
axes[2].axhline(y=0, color="gray", linestyle=":")

axes[0].set_title("CE: train / val  (★ = val min)")
axes[1].set_title("accuracy: train / val")
axes[2].set_title("gap = val CE − train CE")
for ax in axes:
    ax.set_xlabel("iter")
    ax.grid(alpha=0.3)

plt.tight_layout()
fig.legend(loc="upper center", ncol=len(tags) + 1, bbox_to_anchor=(0.5, 1.15))

name = "curve_" + "_".join(tags) + ".png"
if len(tags) == 1:
    out = os.path.join(out_dir, tags[0], name)
else:
    compare_dir = os.path.join(out_dir, "articles")
    os.makedirs(compare_dir, exist_ok=True)
    out = os.path.join(compare_dir, name)
plt.savefig(out, dpi=150, bbox_inches="tight")   # 外に出した凡例が切れないように
print("saved:", out)