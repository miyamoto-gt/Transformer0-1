import common
m = common.build_model()
for b in m.blocks:
    b.attn.mask.fill_(True)
print("mask all True:", all(b.attn.mask.all().item() for b in m.blocks))

common.report_init(m)
common.run(m, tag="no_mask")

print("--- 生成 ---")
print(common.generate(m, n=300))