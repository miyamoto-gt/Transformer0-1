# 平均二乗誤差 MSE を用いた差分
import torch
import torch.nn.functional as F
from base_train import run, V


def loss_fn(logits, y):
    probs = torch.softmax(logits, dim=-1)
    onehot = F.one_hot(y, V).float()
    return ((probs - onehot) ** 2).sum(-1).mean()   # 65文字は合計、位置は平均


if __name__ == "__main__":      # B' から loss_fn だけ import しても学習は走らない
    run(loss_fn, tag="mse")