import torch 
def load_data(path="data/input.txt"):
    text=open(path,encoding="utf-8").read()
    chars=sorted(set(text))
    stoi={c:i for i,c in enumerate(chars)}
    itos={i:c for c,i in stoi.items()}
    data=torch.tensor([stoi[c] for c in text],dtype=torch.long)
    n=int(0.9*len(data))
    return data[:n],data[n:],stoi,itos
def get_batch(data,block_size,batch_size):
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x,y