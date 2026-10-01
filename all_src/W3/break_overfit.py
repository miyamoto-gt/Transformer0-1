import common

SUB = 2000

m = common.build_model()
common.report_init(m)
common.run(m, tag="overfit_2000", 
           train_data=common.train[:SUB])

print("--- 生成 ---")
print(common.generate(m, n=200))