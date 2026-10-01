import matplotlib.pyplot as plt
import common

# 基準
m1 = common.build_model()
h1 = common.run(m1, tag="standard")

# mask なし
m2 = common.build_model()
for b in m2.blocks:
    b.attn.mask.fill_(True)
h2 = common.run(m2, tag="no_mask")

xs = [h["iter"] for h in h1]

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# 左軸: val loss
ax1.plot(xs, [h["val"] for h in h1], color="tab:gray", label="val loss (standard)")
ax1.plot(xs, [h["val"] for h in h2], color="tab:blue", label="val loss (no mask)")
ax1.set_xlabel("iter")
ax1.set_ylabel("val loss")

# 右軸: attn -> t+1
ax2.plot(xs, [h["attn"] for h in h2], color="tab:red", label="attn to t+1 (no mask)")
ax2.axhline(1 / common.block_size, ls=":", color="tab:red", alpha=0.5)
ax2.text(10, 1/common.block_size * 1.5, "uniform (1/128)", color="tab:red", fontsize=8)
ax2.set_ylabel("attention weight to t+1")

# 凡例をまとめる
lines = ax1.get_lines() + [ax2.get_lines()[0]]
ax1.legend(lines, [l.get_label() for l in lines],
           loc="center left", bbox_to_anchor=(0.02, 0.35))

plt.title("W3 #3: loss stays flat while attention grows")
plt.tight_layout()
plt.savefig("docs/compare_attn.png", dpi=150)
plt.close()
print("saved: docs/compare_attn.png")