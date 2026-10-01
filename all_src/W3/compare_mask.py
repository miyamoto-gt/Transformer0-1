import math
import matplotlib.pyplot as plt
import common

ITERS, EVERY = 600, 25

# 基準
print("---standard---")
m1 = common.build_model()
h1 = common.run(m1, tag="standard_600", iters=ITERS, eval_every=EVERY)

# mask なし
print("---no_mask---")
m2 = common.build_model()
for b in m2.blocks:
    b.attn.mask.fill_(True)
h2 = common.run(m2, tag="no_mask_600", iters=ITERS, eval_every=EVERY)

# 重ねて描く
xs = [h["iter"] for h in h1]
plt.plot(xs, [h["val"] for h in h1], label="standard")
plt.plot(xs, [h["val"] for h in h2], label="no mask")
plt.axhline(math.log(common.V), ls="--", color="gray", label="ln(V)")
plt.xlabel("iter"); plt.ylabel("val loss"); plt.legend()
plt.title("W3 #3: causal mask removal")
plt.savefig("docs/compare_mask.png", dpi=150)
plt.close()

print("--- standard ---")
print(common.generate(m1, n=200))
print("--- no mask ---")
print(common.generate(m2, n=200))