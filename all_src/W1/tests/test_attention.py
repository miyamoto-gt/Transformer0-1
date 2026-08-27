import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from attention import softmax, init_weights, attention


def test_shapes():
    rng = np.random.default_rng(0)
    d, T = 8, 4
    x = rng.normal(size=(T, d))
    W_q, W_k, W_v = init_weights(d, rng)

    out, P, scores = attention(x, W_q, W_k, W_v, div=np.sqrt(d))

    assert out.shape == (T, d)
    assert P.shape == (T, T)
    assert scores.shape == (T, T)

    
def test_row_sums_to_one():
    rng = np.random.default_rng(0)
    d, T = 8, 4
    x = rng.normal(size=(T, d))
    W_q, W_k, W_v = init_weights(d, rng)

    _, P, _ = attention(x, W_q, W_k, W_v, div=np.sqrt(d))

    assert np.allclose(P.sum(axis=-1), 1.0)


def test_causal_mask():
    rng = np.random.default_rng(0)
    d, T = 8, 4
    x = rng.normal(size=(T, d))
    W_q, W_k, W_v = init_weights(d, rng)
    mask = np.tril(np.ones((T, T), dtype=bool))

    out_m, P_m, _ = attention(x, W_q, W_k, W_v, div=np.sqrt(d), mask=mask)
    out_n, _, _ = attention(x, W_q, W_k, W_v, div=np.sqrt(d))

    # 上三角が 0
    assert np.allclose(P_m[np.triu_indices(T, k=1)], 0.0)
    # マスク後も行和は 1
    assert np.allclose(P_m.sum(axis=-1), 1.0)
    # 出力が実際に変わっている
    assert not np.allclose(out_m, out_n)


def test_fully_masked_row_produces_nan():
    """全マスク行は softmax で 0/0 になり nan。仕様として固定する。"""
    rng = np.random.default_rng(0)
    d, T = 8, 4
    x = rng.normal(size=(T, d))
    W_q, W_k, W_v = init_weights(d, rng)

    mask = np.tril(np.ones((T, T), dtype=bool))
    mask[0, :] = False          # 0行目を全部塞ぐ

    _, P, _ = attention(x, W_q, W_k, W_v, div=np.sqrt(d), mask=mask)

    assert np.isnan(P[0]).all()
    assert not np.isnan(P[1:]).any()   # 他の行は無事