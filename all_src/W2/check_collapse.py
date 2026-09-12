"""Attention / FFN / 残差 / LayerNorm を1つずつ積み上げて、何が潰れを止めるかを見る。

部品は check_block.py・check_ln_need.py と同じにそろえる
（MultiHeadAttention + nn.Linear の既定init + 出力proj、層ごとに独立した重み）。

測るもの:
  spread … 系列内でトークンがどれだけ散らばっているか（全トークンの平均からの距離）
  rank   … T個のトークンの中に何本の独立な方向が残っているか（最大 T）

深さは24。8層だと残差ありの2条件に差がつかず、
「LayerNormは要らない」と読めてしまう（check_ln_need.py と同じ理由）。
"""
import torch
from attention import MultiHeadAttention
from ffn import FeedForward
from layernorm import LayerNorm

B, T, d, n_head, n_layer = 1, 16, 32, 4, 24
SHOW = {1, 2, 3, 4, 6, 8, 12, 16, 24}

torch.manual_seed(0)
x = torch.randn(B, T, d)


def report(name, use_ffn, use_res, use_ln):
    torch.manual_seed(0)                      # 全条件で同じ重みを使う
    layers = [
        (MultiHeadAttention(d, n_head, block_size=T),
         FeedForward(d), LayerNorm(d), LayerNorm(d))
        for _ in range(n_layer)
    ]

    print(f"--- {name} ---")
    h = x.clone()
    with torch.no_grad():
        for i, (attn, ffn, ln1, ln2) in enumerate(layers):
            a, _, _ = attn(ln1(h) if use_ln else h)
            h = h + a if use_res else a
            if use_ffn:
                f = ffn(ln2(h) if use_ln else h)
                h = h + f if use_res else f
            if i + 1 in SHOW:
                spread = (h - h.mean(dim=1, keepdim=True)).norm(dim=-1).mean()
                rank = torch.linalg.matrix_rank(h[0], rtol=1e-4).item()
                print(f"depth {i+1:2d}: spread={spread:9.6f} rank={rank}")


report("attention only",         use_ffn=False, use_res=False, use_ln=False)
report("attention + ffn",        use_ffn=True,  use_res=False, use_ln=False)
report("+ residual",             use_ffn=True,  use_res=True,  use_ln=False)
report("+ residual + layernorm", use_ffn=True,  use_res=True,  use_ln=True)
