from img_model import ImgMini
from data_img import get_batch
import torch
import os, csv,sys

img_size, patch_size, in_ch, d_model, n_layer=28,7,1,64,4
batch_size = 256
lr = 1e-3
seed=0
torch.manual_seed(seed)
m=ImgMini(img_size, patch_size, in_ch, d_model, n_layer)
opt = torch.optim.AdamW(m.parameters(), lr=lr)
print("params:", sum(p.numel() for p in m.parameters()))
@torch.no_grad()
def evaluate(m, batches):  
    m.eval()
    acc,losses=[],[]
    for (x,y) in batches:
        logits, loss = m(x, y)
        acc.append((logits.argmax(-1) == y).float().mean().item())
        losses.append(loss.item())
    m.train()
    return sum(losses) / len(losses), sum(acc) / len(acc)

max_iters=500
rows=[]
eval_gen=torch.Generator()
eval_gen.manual_seed(seed+2)
train_eval_batches = [get_batch("train", batch_size,generator=eval_gen)for i in range(20)]
val_eval_batches = [get_batch("val", batch_size,generator=eval_gen)for i in range(20)]

print(f"{'iter':>5} | {'train CE':>8} | {'train acc':>9} | {'val CE':>7} | {'val acc':>7}")

for it in range(max_iters):  
    if it % 100 == 0 or it == max_iters - 1:
        tce,tacc=evaluate(m, train_eval_batches)
        vce,vacc=evaluate(m, val_eval_batches)
        rows.append([it, tce,tacc, vce, vacc])
        
        print(f"{it:>5} | {tce:>8.4f} | {tacc:>9.1%} | {vce:>7.4f} | {vacc:>7.1%}")
    x, y = get_batch("train", batch_size)
    _, loss = m(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
tag=(sys.argv[1] if len(sys.argv) > 1 else "base")
current_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(current_dir, "temp")
os.makedirs(out_dir, exist_ok=True)
run_dir=os.path.join(out_dir,tag)
os.makedirs(run_dir,exist_ok=True)
with open(os.path.join(run_dir, f"log_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "train_ce", "train_acc", "val_ce", "val_acc"])
        w.writerows(rows)
    
        

