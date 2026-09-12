"""LayerNorm がないと何が壊れるかを見る。

部品は check_block.py と同じにそろえる（MultiHeadAttention + nn.Linear の既定init +
出力proj、層ごとに独立した重み）。測りたいのは LayerNorm の有無なので、
それ以外は本番と同じものを使わないと、別のものを測ることになる。

以前はここで手書きの簡易attentionを使っていた。初期化スケールが
  手書き  torch.randn(d,d)/d**0.5  → 正規分布なので std = 1/√d   = 0.171
  本物    nn.Linear の既定         → 一様分布なので std = 1/√(3d) = 0.102
と1.7倍違い、さらに出力projが無いぶん枝の出力が大きくなる。
その結果、LayerNorm ありでも std が12層で10倍に膨らんで見えていた。
実際は本物の部品なら12層で1.5倍しか動かない（末尾の比較を参照）。

P.max ではなく「最終行の P.max」を見ているのは、causal mask があると
0行目は必ず [1,0,...,0] になり、行列全体の max が常に 1.0 に張り付くため。
"""
import torch
from attention import MultiHeadAttention
from ffn import FeedForward
from layernorm import LayerNorm

B, T, d, n_head, n_layer = 1, 16, 32, 4, 24

torch.manual_seed(0)
x = torch.randn(B, T, d)

for use_ln in [False, True]:
    torch.manual_seed(0)                      # 両条件でまったく同じ重みを使う
    layers = [
        (MultiHeadAttention(d, n_head, block_size=T),
         FeedForward(d), LayerNorm(d), LayerNorm(d))
        for _ in range(n_layer)
    ]

    print(f"--- LayerNorm: {use_ln} ---")
    h = x.clone()
    with torch.no_grad():
        for i, (attn, ffn, ln1, ln2) in enumerate(layers):
            a, P, _ = attn(ln1(h) if use_ln else h)
            h = h + a
            h = h + ffn(ln2(h) if use_ln else h)
            if (i + 1) % 4 == 0 or i == 0:
                last_row = P[0, :, -1, :]      # 全文脈が見える行だけ見る
                print(f"depth {i+1:2d}: h.std={h.std():9.3f}  "
                      f"最終行P.max={last_row.max().item():.4f}")

print()
print("--- 枝の出力スケール（std=1 の入力を1回通したとき）---")
torch.manual_seed(0)
u = torch.randn(1, T, d)
Wq, Wk, Wv = (torch.randn(d, d) / d**0.5 for _ in range(3))


def toy_attention(h):                         # 以前ここで使っていた手書き版
    q, k, v = h @ Wq, h @ Wk, h @ Wv
    P = torch.softmax(q @ k.transpose(-2, -1) / d**0.5, dim=-1)
    return P @ v


torch.manual_seed(0)
attn = MultiHeadAttention(d, n_head, block_size=T)
with torch.no_grad():
    print(f"手書きattention(projなし) out.std = {toy_attention(u).std():.3f}")
    print(f"本物MultiHead(projあり)   out.std = {attn(u)[0].std():.3f}")
    print(f"  手書きinitの重みstd = {Wv.std():.4f}  (1/sqrt(d)    = {1/d**0.5:.4f})")
    print(f"  nn.Linear既定の重みstd = {attn.qkv.weight.std():.4f}  (1/sqrt(3d) = {1/(3*d)**0.5:.4f})")
