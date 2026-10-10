# 実験1-B': MSE + 学習率を大きくする(仮説「スケールを補えば MSE も活躍できる」の検証)

from base_train import run, lr
from exp_mse import loss_fn   # B と同じ採点方法をそのまま使う
 
if __name__ == "__main__":
    run(loss_fn, tag="mse_lr*10", lr=lr * 10)