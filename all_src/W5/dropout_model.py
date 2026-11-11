import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
W2_DIR = os.path.abspath(os.path.join(current_dir, "..", "W2"))
sys.path.insert(0, W2_DIR)

import torch
import torch.nn as nn
from model import GPTMini


class DropoutModel(GPTMini):
    def __init__(self,V,d_model,block_size,n_layer,dropout):
        super().__init__(V,d_model,block_size,n_layer)
        self.drop=nn.Dropout(dropout)
    def forward(self, idx, targets=None):
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device)
            x = self.tok_emb(idx) + self.pos_emb(pos)      # (B,T,d)
            x=self.drop(x)
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
    m = DropoutModel(65, 64, 32, 4, 0.5)
    idx = torch.randint(0, 65, (2, 32))      # 入力(B=2, T=32)

    m.train()
    a, _ = m(idx)
    b, _ = m(idx)
    print("train:", torch.equal(a, b))                     # a と b が完全に同じか

    m.eval()
    c, _ = m(idx)
    d, _ = m(idx)
    print("eval :", torch.equal(c,d))                     # c と d が完全に同じか