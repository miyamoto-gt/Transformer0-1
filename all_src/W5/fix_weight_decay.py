# 重みの減衰
import torch.nn.functional as F
from base_train import run,V
def loss_fn(logits, y):
    return F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
 
if __name__ == "__main__":
    run(loss_fn,tag="wd0.1_d128",train_frac=0.1,d_model=128,weight_decay=0.1)