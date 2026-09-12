import torch
from block import Block
from layernorm import LayerNorm

class GPTMini(torch.nn.Module):
    def __init__(self, vocab_size, d_model, block_size, n_layer):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model)
        self.pos_emb = torch.nn.Embedding(block_size, d_model)
        self.blocks = torch.nn.ModuleList(
            [Block(d_model, block_size) for _ in range(n_layer)]
        )
        self.ln_f = LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)      # (B,T,d)
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