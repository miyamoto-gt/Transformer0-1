import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
W2_DIR = os.path.abspath(os.path.join(current_dir, "..", "W2"))
sys.path.insert(0, W2_DIR)

import torch
import torch.nn as nn
from model import GPTMini

class FeatureModel(GPTMini):
    def __init__(self,V,d_model,block_size,n_layer,num_intervals):
        super().__init__(V,d_model,block_size,n_layer)
        self.num_intervals=num_intervals
        with torch.random.fork_rng():# 乱数シードを固定して初期化
            self.interval_emb=nn.Embedding(num_intervals,d_model)
    def forward(self, idx, targets=None,interval=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)   # (B,T,d)
        if interval is not None:
            e=self.interval_emb(interval)
            x=x+e.unsqueeze(1)  # (B,T,d)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                        # (B,T,vocab)
        if targets is None:
            return logits, None
        loss = torch.nn.functional.cross_entropy(
            logits.view(B * T, -1), targets.view(B * T)
        )
        return logits, loss
if __name__ == "__main__":
    torch.manual_seed(0)
    base = GPTMini(65, 64, 32, 4)
    print("GPTMini     :", torch.randint(100000, (5,)))

    torch.manual_seed(0)
    rand = FeatureModel(65, 64, 32, 4, 112)
    print("FeatureModel:", torch.randint(100000, (5,)))
