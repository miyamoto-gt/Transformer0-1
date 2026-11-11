"""W5 基準ファイル: 学習・評価・保存の共通部分
実験ファイルは run() の引数を1つだけ変えて呼ぶ(損失はCE固定)。
"""
import os, sys, csv

current_dir = os.path.dirname(os.path.abspath(__file__))
W2_DIR = os.path.abspath(os.path.join(current_dir, "..", "W2"))
sys.path.insert(0, W2_DIR)

import torch
import torch.nn.functional as F
from model import GPTMini
from data_utils import load_data
from feat_model import FeatureModel
from dropout_model import DropoutModel


# ---- 固定する設定(W2 と同じ値にそろえる) --------------------------------
lr = 1e-3           # ← W2 の train.py の学習率に合わせる
block_size, batch_size =32, 32
eval_iters=50
seed = 0

out_dir = os.path.join(current_dir, "temp")
os.makedirs(out_dir, exist_ok=True)

train, val, stoi, itos = load_data(os.path.join(W2_DIR, "data", "input.txt"))
V = len(stoi)


def get_batch(data,generator=None):
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y,ix

def make_eval_batches(data,seed):
    generator= torch.Generator()
    generator.manual_seed(seed)
    batches=[]
    for _ in range(eval_iters):
        batches.append(get_batch(data,generator=generator))
    return batches

@torch.no_grad()
def evaluate(m,batches,intervals=None):
    """共通の物差し: 渡した評価バッチの CE と正解率を測る"""
    m.eval()
    ces, accs = [], []
    for i,(x,y,ix) in enumerate(batches):
        if intervals is None:
            logits, _ = m(x)
        else:
            logits, _ = m(x,interval=intervals[i])                                  # targets なし → (B, T, V)
        ces.append(F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).item())
        accs.append((logits.argmax(-1) == y).float().mean().item())
    m.train()
    return sum(ces) / len(ces), sum(accs) / len(accs)


def run(loss_fn, tag, lr=lr,weight_decay=0,d_model=64,n_layer=4,max_iters=3000,eval_interval=150,
        train_frac=1.0,interval_width=None,interval_mode="pos",patience=None,min_delta=0.02,dropout=None):
    count =0
    best_iter=0
    min_loss = float("inf")
    train_range=int(len(train)*train_frac)
    train_data=train[:train_range]
    print("Train data size:", len(train_data))
    run_dir=os.path.join(out_dir,tag)
    os.makedirs(run_dir,exist_ok=True)
    train_eval_batches=make_eval_batches(train_data, seed)
    val_eval_batches=make_eval_batches(val, seed)
    num_interval = None
    if interval_width is not None:
        num_interval=max(len(train_data),len(val))//interval_width+1 #区間IDが０から数え始めるため、０からカウント
    if interval_width is None:
        train_eval_intervals = None
        val_eval_intervals = None
    elif interval_mode == "pos":
        train_eval_intervals = [ix // interval_width for x, y, ix in train_eval_batches]
        val_eval_intervals   = [ix // interval_width for x, y, ix in val_eval_batches]
    else:
        eval_noise = torch.Generator()
        eval_noise.manual_seed(seed+1)
        train_eval_intervals = [torch.randint(0,num_interval, (batch_size,),generator=eval_noise) for x, y, ix in train_eval_batches]
        val_eval_intervals   = [torch.randint(0,num_interval, (batch_size,),generator=eval_noise) for x, y, ix in val_eval_batches]
    torch.manual_seed(seed)
    if interval_width is not None:
        m= FeatureModel(V,d_model,block_size,n_layer,num_interval)
    elif dropout is not None:
        m=DropoutModel(V,d_model,block_size,n_layer,dropout)
    else:
        m=GPTMini(V, d_model, block_size, n_layer)
    interval_train=torch.Generator()
    interval_train.manual_seed(seed+2)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)  # adamW で学習するのが W2 の train.py と同じ
    print("weight_decay:",opt.param_groups[0]['weight_decay'])  # weight_decay の値を確認する
    rows = []
    print(f"[{tag}] {'iter':>5} | {'train CE':>8} | {'train acc':>9} | {'val CE':>7} | {'val acc':>7} | {'lm_head勾配':>10}")
    for it in range(max_iters + 1): 
        x, y,ix= get_batch(train_data)
        if interval_width is None:
            logits, _ = m(x)
        elif interval_mode=="pos":
            interval=ix//interval_width
            logits, _ = m(x,interval=interval) 
        else:
            interval=torch.randint(0,num_interval, (batch_size,),generator=interval_train)
            logits, _ = m(x,interval=interval)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()

        if it % eval_interval== 0:
            gnorm = m.lm_head.weight.grad.norm().item()  
            tce,tacc=evaluate(m,train_eval_batches,train_eval_intervals)
            vce,vacc=evaluate(m,val_eval_batches,val_eval_intervals)
            rows.append([it, tce,tacc, vce, vacc, gnorm])
            print(f"[{tag}] {it:>5} | {tce:>8.4f} | {tacc:>7.3f} | {vce:>7.4f} | {vacc:>7.3f} | {gnorm:>10.2e}")
            min_loss = min(min_loss,vce)
            if vce==min_loss:
                count=0
                best_iter=it
                torch.save(m.state_dict(), os.path.join(run_dir, f"ckpt_best_{tag}.pt"))
            else:
                count+=1
            print(f"{best_iter:>5}iter | min_loss:{min_loss:>.4f} | count:{count:>1}")
            if patience is not None and count>=patience and vce-min_loss >= min_delta:
                break
        opt.step()
    
    
    with open(os.path.join(run_dir, f"log_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "train_ce", "train_acc", "val_ce", "val_acc", "lm_head_grad"])
        w.writerows(rows)
    torch.save(m.state_dict(), os.path.join(run_dir, f"ckpt_{tag}.pt"))

    pt_contents=torch.load(os.path.join(run_dir, f"ckpt_best_{tag}.pt"))
    m.load_state_dict(pt_contents)
    bce, bacc=evaluate(m, val_eval_batches, val_eval_intervals)
    print(bce)
    # 生成サンプル(T=1.0)
    m.eval()
    torch.manual_seed(0)
    idx = torch.zeros((1, 1), dtype=torch.long)
    with torch.no_grad():
        for _ in range(500):
            logits, _ = m(idx[:, -block_size:])
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
    with open(os.path.join(run_dir, f"sample_{tag}.txt"), "w") as f:
        f.write("".join(itos[i.item()] for i in idx[0]))