from base_train import make_eval_batches,val
import torch

if __name__ == "__main__":
    #評価バッチの準備が再現性を持つことの確認
    t1=make_eval_batches(val, 0)
    t2=make_eval_batches(val, 0)
    print(torch.equal(t1[0][0],t2[0][0])) 
    print(torch.equal(t1[0][1],t2[0][1]))  

    #評価バッチの準備が学習の乱数に影響しないことの確認
    s1=torch.get_rng_state()
    s2=torch.get_rng_state()
    print(torch.equal(s1,s2))


