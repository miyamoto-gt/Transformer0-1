import numpy as np


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def init_weights(d, rng, scale=True):
    s = np.sqrt(d) if scale else 1.0
    return [rng.normal(size=(d, d)) / s for _ in range(3)]


def attention(x, W_q, W_k, W_v, div=None,mask=None):
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    scores = Q @ K.T
    if div is not None:
        scores = scores / div
    if mask is not None:
        scores=np.where(mask,scores,-np.inf)


    P = softmax(scores)
    return P @ V, P, scores


def compare_divisors(d, rng, scale=True):
    x = rng.normal(size=(4, d))
    W_q, W_k, W_v = init_weights(d, rng, scale=scale)

    _, _, s = attention(x, W_q, W_k, W_v, div=None)
    print(f"--- d={d}, scale={scale} ---")
    print(f"  scores std: {s.std():.2f}")

    for name, div in [("raw", None), ("/sqrt(d)", np.sqrt(d)), ("/d", d)]:
        _, P, _ = attention(x, W_q, W_k, W_v, div=div)
        print(f"  {name:10s} P.max={P.max(axis=-1).mean():.4f}  "
              f"sum={P.sum(axis=-1).round(6)}")


if __name__ == "__main__":
    print("##################")
    print("----attention----")
    print("##################\n")
    rng = np.random.default_rng(0)
    compare_divisors(8, rng)
    compare_divisors(64, rng)
    compare_divisors(64, rng, scale=False)   
    rng = np.random.default_rng(0)

    #mask check
    print("##################")
    print("----add mask----")
    print("##################")
    d = 8
    x = rng.normal(size=(4, d))
    W_q, W_k, W_v = init_weights(d, rng)
    mask = np.tril(np.ones((4, 4), dtype=bool))

    out_m, P_m, _ = attention(x, W_q, W_k, W_v, div=np.sqrt(d), mask=mask)
    out_n, P_n, _ = attention(x, W_q, W_k, W_v, div=np.sqrt(d))

    print(P_m.round(3))
    print("row sums:", P_m.sum(axis=-1))
    print("out changed:", not np.allclose(out_m, out_n))