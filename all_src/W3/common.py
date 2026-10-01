import os, sys
import math                                   
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
W2_DIR = os.path.abspath(os.path.join(current_dir, '..', 'W2'))
sys.path.insert(0, W2_DIR)

import torch
import matplotlib.pyplot as plt               
from model import GPTMini
from data_utils import load_data, get_batch


block_size, batch_size = 128, 32
d_model, n_layer = 64, 4
max_iters, eval_interval, eval_iters = 600, 25, 20
lr = 1e-3

DATA = os.path.join(W2_DIR, "data", "input.txt")
train, val, stoi, itos = load_data(DATA)
V = len(stoi)

def build_model():
    torch.manual_seed(1337)
    return GPTMini(V, d_model, block_size, n_layer)

@torch.no_grad()#関数内で勾配計算を行わない
def estimate_loss(m,train_data=None):
    d=train_data if train_data is not None else train
    m.eval()
    out = {}
    for name, data in [("train", d), ("val", val)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size)
            _, loss = m(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    m.train()
    return out

@torch.no_grad()
def attn_to_next(m, layer=0):
    x, _ = get_batch(val, block_size, batch_size)
    h = m.tok_emb(x) + m.pos_emb(torch.arange(block_size))
    b = m.blocks[layer]
    _, P, _ = b.attn(b.ln1(h))
    T = block_size
    diag = P[:, :, torch.arange(T - 1), torch.arange(1, T)]
    return diag.mean().item()

@torch.no_grad()
def measure_init(m):
    m.eval()
    losses = torch.zeros(eval_iters)
    stds   = torch.zeros(eval_iters)
    maxps  = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(val, block_size, batch_size)
        logits, loss = m(x, y)
        losses[k] =loss.item()
        stds[k]   = logits.std().item()
        maxps[k]  = logits.softmax(-1).max(-1).values.mean().item()
    m.train()
    return losses.mean().item(), stds.mean().item(), maxps.mean().item()

def report_init(m):
    loss, std, maxp = measure_init(m)
    print(f"init loss     : {loss:.4f}   (ln(V) = {math.log(V):.4f})")
    print(f"logits std    : {std:.4f}")
    print(f"max prob mean : {maxp:.4f}   (1/V = {1/V:.4f})")
    return loss, std, maxp

def block_grad_norm(b):
    sq = sum(p.grad.pow(2).sum() for p in b.parameters() if p.grad is not None)
    return sq.sqrt().item()

@torch.no_grad()
def generate(m,n=300,start=0):
    m.eval()
    idx = torch.full((1,1),start,dtype=torch.long)
    for _ in range(n):
        logits,_=m(idx[:,-block_size:])
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, nxt), dim=1)
    m.train()
    return "".join(itos[i.item()] for i in idx[0])

def run(m,tag,lr_override=None,shuffle_y=False,
        iters=max_iters, eval_every=eval_interval,train_data=None):
    data=train_data if train_data is not None else train
    history = []                                  
    t0 = time.time()
    opt = torch.optim.AdamW(m.parameters(), lr=lr_override or lr)
    for it in range(iters + 1):
        if it % eval_every == 0:
            l = estimate_loss(m,train_data)
            a=attn_to_next(m)
            print(f"iter {it:5d}  train {l['train']:.4f}  validation {l['val']:.4f}  attn {a:.4f}")
            history.append({"iter": it, "train": l["train"], "val": l["val"], "attn": a})  
        x, y = get_batch(data, block_size, batch_size)
        if shuffle_y:
            y=y[torch.randperm(y.size(0))]# y をシャッフル
        _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    elapsed = time.time() - t0
    print(f"elapsed: {elapsed:.2f} sec")
    docs_dir = os.path.join(current_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)     
    
    #--- グラフ ---
    xs = [h["iter"] for h in history]
    plt.plot(xs, [h["train"] for h in history], label="train")
    plt.plot(xs, [h["val"] for h in history], label="val")     
    plt.axhline(math.log(V), ls="--", color="gray", label="ln(V)")
    plt.xlabel("iter"); plt.ylabel("loss"); plt.legend()
    out_path = os.path.join(docs_dir, f"{tag}.png")
    plt.title(f"W3 {tag} ({iters} iter)")
    plt.savefig(out_path, dpi=150)
    plt.close()                                   
    print("saved:", out_path)
    return history


