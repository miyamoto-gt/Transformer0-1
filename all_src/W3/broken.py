import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
W2_DIR = os.path.abspath(os.path.join(current_dir, '..', 'W2'))
sys.path.insert(0, W2_DIR)

import torch
from block import Block

class BlockNoResidual(Block):
    def forward(self,x):
        a,_,_ =self.attn(self.ln1(x))
        x=a
        x=self.ffn(self.ln2(x))
        return x