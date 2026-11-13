import os, sys, csv
import torch.nn as nn
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
W2_DIR = os.path.abspath(os.path.join(current_dir, "..", "W2"))
sys.path.insert(0, W2_DIR)

from block import Block
from layernorm import LayerNorm
from patch_embed import Embed

class ImgMini(nn.Module):
    def __init__(self,img_size, patch_size, in_ch, d_model, n_layer, n_class=10):
        super().__init__()
        self.patch_emb=Embed(patch_size=patch_size, in_ch=in_ch, d_model=d_model, img_size=img_size)
        n_patches = self.patch_emb.n_patches
        self.pos_emb=nn.Embedding(n_patches, d_model)
        self.blocks=nn.ModuleList([Block(d_model, n_patches) for _ in range(n_layer)])
        self.ln_f=LayerNorm(d_model)
        self.head=nn.Linear(d_model, n_class)


    def forward(self, x,targets=None):
            x = self.patch_emb(x)
            patches=self.patchify(x)
            self.pos_emb = torch.arange(patches, device=x.device)
            for blk in self.blocks:
                self.ln_f
                x[:, -1]
                self.head
                if  targets is None:

                else:
                     F.cross_entropy(logits,targets)
                     

            
if __name__ == "__main__":
    model = ImgMini(28, 7, 1, 64, n_layer=4)
    print(sum(p.numel() for p in model.parameters()))