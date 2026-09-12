"""W2の不変条件を固定する。

check_*.py は「数字を見る」ためのもの（人間が読んで判断する）。
このテストは「壊れたら気づく」ためのもの（機械が判定する）。
Week 3 で意図的にモデルを壊すとき、意図した破壊と事故を切り分ける土台になる。
"""
import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from attention import Head, MultiHeadAttention
from block import Block
from ffn import FeedForward
from layernorm import LayerNorm
from model import GPTMini

D, T, BLOCK = 32, 6, 16


# ---------- LayerNorm ----------

def test_layernorm_normalizes_last_dim():
    """各トークンについて平均0・分散1にする。バッチや位置には依存しない。"""
    ln = LayerNorm(D)
    x = torch.randn(2, T, D) * 5 + 3          # わざとスケールと平均をずらす
    out = ln(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, T), atol=1e-5)
    assert torch.allclose(out.std(dim=-1, unbiased=False), torch.ones(2, T), atol=1e-3)


def test_layernorm_matches_pytorch():
    """自前実装が torch.nn.LayerNorm と一致する。"""
    mine, ref = LayerNorm(D), torch.nn.LayerNorm(D)
    with torch.no_grad():
        mine.gamma.copy_(ref.weight)
        mine.beta.copy_(ref.bias)
    x = torch.randn(2, T, D)
    assert torch.allclose(mine(x), ref(x), atol=1e-5)


# ---------- FFN ----------

def test_ffn_is_position_wise():
    """FFNは各位置を独立に処理する。並べ替えてから通しても、通してから並べ替えても同じ。"""
    ff = FeedForward(D)
    x = torch.randn(2, T, D)
    perm = torch.randperm(T)
    assert torch.allclose(ff(x)[:, perm], ff(x[:, perm]), atol=1e-6)


def test_ffn_is_nonlinear():
    """clamp(min=0) が効いていることを2階差分で確かめる。

    f(a+b) - (f(a)+f(b)) は bias の分が残るだけなので、ReLU を外しても 0 にならない。
    アフィン写像 f(x)=Ax+c なら f(a+b)-f(a)-f(b)+f(0) が厳密に 0 になる。
    """
    ff = FeedForward(D)
    a, b = torch.randn(1, 1, D), torch.randn(1, 1, D)
    z = torch.zeros(1, 1, D)
    second_diff = ff(a + b) - ff(a) - ff(b) + ff(z)
    assert second_diff.abs().max() > 1e-3


def test_ffn_without_relu_would_be_affine():
    """上のテストが本当に非線形性を見ていることの裏取り。

    ReLU を通さない経路は 2階差分が厳密に 0 になる。
    ここが 0 にならなければ、test_ffn_is_nonlinear は別のものを測っている。
    """
    ff = FeedForward(D)
    affine = lambda x: ff.w2(ff.w1(x))          # clamp を挟まない＝アフィン
    a, b = torch.randn(1, 1, D), torch.randn(1, 1, D)
    z = torch.zeros(1, 1, D)
    second_diff = affine(a + b) - affine(a) - affine(b) + affine(z)
    assert second_diff.abs().max() < 1e-5


# ---------- Attention ----------

@pytest.mark.parametrize("module", ["head", "multihead"])
def test_attention_is_causal_and_normalized(module):
    """未来を見ない（上三角が0）かつ行和が1。"""
    m = Head(D, D, BLOCK) if module == "head" else MultiHeadAttention(D, 4, BLOCK)
    _, P, _ = m(torch.randn(2, T, D))
    upper = P[..., :, :].triu(1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-7)
    assert torch.allclose(P.sum(dim=-1), torch.ones_like(P.sum(dim=-1)), atol=1e-5)


def test_attention_accepts_shorter_than_block_size():
    """mask を [:T,:T] で切っているので T < block_size でも動く。"""
    m = MultiHeadAttention(D, 4, BLOCK)
    for t in (1, 3, BLOCK):
        out, _, _ = m(torch.randn(1, t, D))
        assert out.shape == (1, t, D)


def test_multihead_param_count_is_independent_of_n_head():
    """n_head を変えてもパラメータ数は変わらない。

    記事4章の「ヘッド数だけを変えた公平な比較」という主張がここに依存している。
    """
    counts = {
        nh: sum(p.numel() for p in MultiHeadAttention(D, nh, BLOCK).parameters())
        for nh in (1, 2, 4, 8)
    }
    assert len(set(counts.values())) == 1, counts


# ---------- Block ----------

@pytest.mark.parametrize("pre_ln", [True, False])
def test_block_preserves_shape(pre_ln):
    blk = Block(D, BLOCK, n_head=4, pre_ln=pre_ln)
    x = torch.randn(2, T, D)
    assert blk(x).shape == x.shape


def test_block_is_causal():
    """t以降を書き換えても、t未満の出力は変わらない。"""
    blk = Block(D, BLOCK, n_head=4)
    x = torch.randn(2, T, D)
    x2 = x.clone()
    x2[:, 3:, :] = torch.randn(2, T - 3, D)
    with torch.no_grad():
        assert torch.allclose(blk(x)[:, :3], blk(x2)[:, :3], atol=1e-6)


def test_pre_ln_gradient_reaches_input_but_post_ln_does_not():
    """記事2章の主張を固定する。

    Pre-LN は残差の本流に手を入れないので勾配が素通りする。
    Post-LN は毎層 LayerNorm を挟むので、12層で勾配が消える。
    """
    grads = {}
    for pre in (True, False):
        torch.manual_seed(0)
        blocks = torch.nn.ModuleList(
            [Block(D, BLOCK, n_head=4, pre_ln=pre) for _ in range(12)]
        )
        x = torch.randn(1, T, D, requires_grad=True)
        h = x
        for blk in blocks:
            h = blk(h)
        h.sum().backward()
        grads[pre] = x.grad.norm().item()

    assert grads[True] > 1.0, grads          # Pre-LN: 勾配が生きている
    assert grads[False] < 1e-5, grads        # Post-LN: 消えている
    assert grads[True] / grads[False] > 1e6, grads


# ---------- GPTMini ----------

def test_model_shapes_and_loss():
    vocab = 65
    m = GPTMini(vocab, D, BLOCK, n_layer=2)
    idx = torch.randint(vocab, (2, T))
    logits, loss = m(idx, targets=idx)
    assert logits.shape == (2, T, vocab)
    assert loss.ndim == 0


def test_model_loss_at_init_is_near_log_vocab():
    """学習前のlossは ln(語彙数)。ここがずれていたら配線が間違っている。"""
    vocab = 65
    torch.manual_seed(0)
    m = GPTMini(vocab, D, BLOCK, n_layer=2)
    idx = torch.randint(vocab, (8, BLOCK))
    _, loss = m(idx, targets=idx)
    assert abs(loss.item() - math.log(vocab)) < 0.5, loss.item()


def test_model_is_causal():
    """位置kのトークンを変えても、位置k未満のlogitsは変わらない。

    これが壊れると「未来を見ている」状態になり、lossだけが不当に下がる。
    数字を見ても気づけない種類の事故なので、テストで止める。
    """
    vocab = 65
    torch.manual_seed(0)
    m = GPTMini(vocab, D, BLOCK, n_layer=2)
    m.eval()
    idx = torch.randint(vocab, (1, BLOCK))
    k = BLOCK // 2
    idx2 = idx.clone()
    idx2[0, k] = (idx[0, k] + 1) % vocab
    with torch.no_grad():
        a, _ = m(idx)
        b, _ = m(idx2)
    assert torch.allclose(a[:, :k], b[:, :k], atol=1e-6)
    assert not torch.allclose(a[:, k:], b[:, k:], atol=1e-6)


def test_model_handles_variable_length():
    vocab = 65
    m = GPTMini(vocab, D, BLOCK, n_layer=2)
    for t in (1, 5, BLOCK):
        logits, _ = m(torch.randint(vocab, (1, t)))
        assert logits.shape == (1, t, vocab)


def test_head_matches_reference_formula():
    """Head の出力が softmax(QK^T/√d_k)V と厳密に一致する。

    スケーリング係数を落とす／変えるといった変更をここで止める。
    W1 で確かめた「√d_k で割る」がコードから消えていないことの担保。
    """
    torch.manual_seed(0)
    h = Head(D, D, BLOCK)
    x = torch.randn(1, T, D)

    with torch.no_grad():
        out, P, scores = h(x)
        q, k, v = h.query(x), h.key(x), h.value(x)
        ref = q @ k.transpose(-2, -1) * (D ** -0.5)
        ref = ref.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
        ref_P = torch.softmax(ref, dim=-1)

    assert torch.allclose(P, ref_P, atol=1e-6)
    assert torch.allclose(out, ref_P @ v, atol=1e-6)


def test_stacked_blocks_keep_full_rank():
    """Blockを8層積んでもトークンが1点に潰れない。

    記事1章の「残差がrankを守る」に対応する。残差を外すと
    Attentionの平均化でrankが1に落ちるので、ここで検出できる。
    """
    torch.manual_seed(0)
    n_tok = 16
    h = torch.randn(1, n_tok, D)
    with torch.no_grad():
        for _ in range(8):
            h = Block(D, block_size=n_tok, n_head=4)(h)
    rank = torch.linalg.matrix_rank(h[0], rtol=1e-4).item()
    assert rank == n_tok, f"rank={rank} (期待 {n_tok}): トークンが潰れている"
