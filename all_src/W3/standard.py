import common

m=common.build_model()
common.report_init(m)
common.run(m, tag="standard")

print("--- 生成 ---")
print(common.generate(m, n=300))