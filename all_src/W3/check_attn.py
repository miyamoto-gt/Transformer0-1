import common

m = common.build_model()
print("mask あり:", common.attn_to_next(m))

m2 = common.build_model()
for b in m2.blocks:
    b.attn.mask.fill_(True)
print("mask なし:", common.attn_to_next(m2), "  (1/128 =", 1/128, ")")