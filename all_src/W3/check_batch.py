import torch
import common

x, y = common.get_batch(common.train, common.block_size, common.batch_size)

print("--- 正常 ---")
print("x:", repr("".join(common.itos[i.item()] for i in x[0][:20])))
print("y:", repr("".join(common.itos[i.item()] for i in y[0][:20])))

y_shuf = y[torch.randperm(y.size(0))]
print("--- シャッフル後 ---")
print("x:", repr("".join(common.itos[i.item()] for i in x[0][:20])))
print("y:", repr("".join(common.itos[i.item()] for i in y_shuf[0][:20])))