import torch.nn as nn
import torch
class Embed(nn.Module):
    def __init__(self,patch_size, in_ch, d_model,img_size):
        super().__init__()
        self.m=nn.Linear(in_ch * patch_size * patch_size, d_model)
        self.n_patches=(img_size // patch_size) ** 2
        self.patch_size = patch_size

    def patchify(self,x):

        B, C, H, W= x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0
        x = x.reshape(B, C, H // p, p, W // p, p)
        x=x.permute(0, 2, 4, 1, 3, 5)
        patches = x.reshape(B, -1, C * p * p)
        return patches
    
    def forward(self, x):
        patches=self.patchify(x)
        out = self.m(patches)
        return out

    
if __name__ == "__main__":
    e=Embed(patch_size=7, in_ch=1, d_model=64, img_size=28)
    x = torch.randn(2, 1, 28, 28)
    patches =e.patchify(x)
    out =e(x)
    print(patches.shape, out.shape)
    print(torch.equal(patches[0, 0], x[0, 0, :7, :7].flatten()))     
    print(torch.equal(patches[0, 1], x[0, 0, :7, 7:14].flatten()))   
