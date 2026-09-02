---
title: "numpyでAttentionを実装したら、√dで割るのは万能ではなかった"
emoji: "🔍"
type: "tech"
topics: ["transformer", "numpy", "深層学習", "機械学習"]
published: true
---

# はじめに

TransformerのSelf-Attentionを、計算式から実装した。

Attentionは「文章の中でどの単語がどの単語と深く関係しているか」を計算し、
重要な部分に重みを置いて文脈を捉える仕組みである。

今回は実装を通して次を理解した。

- なぜこの計算式になるのか
- なぜ次元数のルートで割るのか

結論から言うと、√dで割るのは万能ではなかった。

# 変えたこと

- `softmax` / `init_weights` / `attention` をnumpyで実装
- スコアを割る数を raw / √d / d の3通りで比較できるようにした
- 初期化のスケールをOFFにできる `scale` フラグを付けた
- causal maskを追加した

## 実装

実装したのはこの式。

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

```python
def attention(x, W_q, W_k, W_v, div=None, mask=None):
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    scores = Q @ K.T
    if div is not None:
        scores = scores / div
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)   # True=通す
    P = softmax(scores)
    return P @ V, P, scores
```

`div` を引数にしたのは、割る数を変えて比較するため。
`init_weights` の `scale=False` は、初期化をわざと壊すためのフラグ。

```python
def init_weights(d, rng, scale=True):
    s = np.sqrt(d) if scale else 1.0
    return [rng.normal(size=(d, d)) / s for _ in range(3)]
```

## Maskの導入

Maskは未来の単語を参照しないようにするためのものだ。

学習時は文章の全文が手元にあるため、そのままだと各位置から未来の単語まで見えてしまう。
答えを見ながら答えを書くことになり、生成時（未来がまだ存在しない）とは
別のタスクを学習してしまう。

そこで未来にあたる位置を `-inf` にして、softmaxで0にする。

```python
mask = np.tril(np.ones((T, T), dtype=bool))   # 下三角がTrue = 過去と自分だけ通す
```

# 起きたこと

## 割る数を変えるとP.maxが動く

`P.max(axis=-1).mean()` は「各行で最も大きい重みの平均」。
1に近いほど1箇所に集中していることになる。

```
--- d=8, scale=True ---
  scores std: 2.91
  raw        P.max=0.5420
  /sqrt(d)   P.max=0.4250
  /d         P.max=0.3213
--- d=64, scale=True ---
  scores std: 8.46
  raw        P.max=0.9258
  /sqrt(d)   P.max=0.5684
  /d         P.max=0.2914
```

d=64の `raw` は0.93。ほぼ1箇所に張り付いていて、softmaxが飽和している。
`/d` は0.29で、今度は散りすぎ。`/sqrt(d)` の0.57が中間にある。

std に注目すると、d=8で2.91（$\sqrt{8} \approx 2.83$）、d=64で8.46（$\sqrt{64} = 8$）。
理論値の $\sqrt{d}$ とほぼ一致している。

## 初期化を壊すと√dが効かなくなる

`scale=False` で初期化の割り算をやめた場合。

```
--- d=64, scale=False ---
  scores std: 512.03
  raw        P.max=1.0000
  /sqrt(d)   P.max=1.0000
  /d         P.max=0.9991
```

stdが512。`/sqrt(d)` で割ってもP.maxは1.0000のまま動かない。
`/d` でようやく0.9991とわずかに動く程度。

## causal maskの効果

```
[[1.    0.    0.    0.   ]
 [0.995 0.005 0.    0.   ]
 [0.309 0.34  0.351 0.   ]
 [0.216 0.278 0.211 0.295]]
row sums: [1. 1. 1. 1.]
out changed: True
```

上三角が0になっている。0行目は自分しか見えないので必ず `[1,0,0,0]`。
行和は1のまま保たれている。

`out changed: True` も確認した。Pが下三角になっているだけでは不十分で、
出力に反映されていなければマスクが効いたことにならない。

# なぜ

## √dの正体

q, kの各成分が独立で平均0・分散1のとき、内積はd個の項の和になる。

$$
\mathrm{score} = \sum_{i=1}^{d} q_i k_i
\quad\Longrightarrow\quad
\mathrm{Var}(\mathrm{score}) = d
\quad\Longrightarrow\quad
\mathrm{std} = \sqrt{d}
$$

stdが√dなので、√dで割ればstdが1に戻る。
**候補の中で一番良かったのではなく、方程式を解いた答え**だ。

`/d` で割るとstdが $1/\sqrt{d}$ になる。d=64なら0.125。
今度はスコアが潰れてsoftmaxがほぼ一様分布になる。上のP.max=0.29がそれだ。

なぜstd=1を目指すのかというと、softmaxの勾配が次の形をしているため。

$$
\frac{\partial P_i}{\partial s_j} = P_i(\delta_{ij} - P_j)
$$

Pが1か0に寄ると積が0に落ちて勾配が消える。
std≈1ならスコアがおおむね±2〜3に収まり、選択性と勾配が両立する。

## 効かなくなる理由

上の式は $\mathrm{Var}(q) = \mathrm{Var}(k) = 1$ を前提にしている。

初期化で √d の割り算をやめると、Wの各成分の分散が1のまま。
`q = x @ W_q` はd個の項の和なので、

$$
\mathrm{Var}(q) = d
\quad\Longrightarrow\quad
\mathrm{Var}(\mathrm{score}) = d \cdot \mathrm{Var}(q) \cdot \mathrm{Var}(k) = d^3
\quad\Longrightarrow\quad
\mathrm{std} = d^{1.5}
$$

d=64なら $64^{1.5} = 512$。実測の512.03と一致した。

この状態で√dで割っても、stdは $512 / 8 = 64$ のまま。
expの指数が±64では完全に飽和し、P.maxは1.0に張り付く。

**√dは「初期化が単位分散を保っている」という前提とセットの補正**で、
単独では機能しない。この2つは「分散を1に保つ」という同じ設計思想の
上流と下流にあたる。

## maskをsoftmaxの前に置く理由

softmaxの後に0で埋める実装も書けるが、それだと壊れる。

```
scores = [2, 1, 1.5, 0.5]
P      = [0.36, 0.13, 0.22, 0.08]   ← 和 = 1
0埋め   = [0.36, 0.13, 0, 0]         ← 和 = 0.49
```

行和が1でなくなる。さらに捨てた0.30分の大きさは未来のスコアの大きさで決まるため、
**消したはずの未来が、残った重みの比率に漏れる**。

`-inf` を先に入れれば、$\exp(-\infty) = 0$ となって分母の和にも入らない。
見える位置だけで重みを分け合うので、和は1のまま保たれる。

# つまずいたところ

`np.where` の引数順を2回間違えた。

```python
np.where(条件, Trueのときの値, Falseのときの値)
```

正しくは `np.where(mask, scores, -np.inf)`。
`np.where(scores, mask, -np.inf)` と書くと条件がfloatになり、
「scoresが0以外の位置にTrue/Falseを入れる」という全く別の処理になる。
エラーにならず動いてしまうので気づきにくい。

また `if mask:` と書くとnumpy配列で例外が出る。

```
ValueError: The truth value of an array with more than one element is ambiguous.
```

`if mask is not None:` が正しい。

# まとめ

- √dは $\mathrm{Var}(\mathrm{score}) = d$ を1に戻すための割り算で、解析的に導ける
- ただし初期化が単位分散という前提が崩れると $\mathrm{Var}(\mathrm{score}) = d^3$ になり、√dでは救えない
- maskは正則化ではなく、学習時の条件を生成時に揃えるための制約
- maskはsoftmaxの前に入れる。後だと行和が1でなくなり、未来の情報が漏れる

# 次回

W2で小さなTransformerを組む。lossが下がることを確認するところまで。

コード全体: https://github.com/miyamoto-gt/Transformer0-1


# 参考文献
Scaled Dot-Product Attentionとは 
https://zenn.dev/yuto_mo/articles/72c07b702c50df