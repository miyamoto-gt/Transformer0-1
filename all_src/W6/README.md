# 第６週

W6 の目標は「連続的なものをモデルの入力形式に変換する」を体験すること。
胴体(W2 の `Block`)は変えず、入口だけを差し替える。

- 画像:MNIST をパッチに切ってトークンにし、数字を分類する
- 文字:同じ文を3つのトークナイザで区切り、トークン数と区切り方を比べる

```
W2:  文字ID → 埋め込み表       → [Block × 4] → 次の文字(65クラス)
W6:  画像   → パッチ + Linear  → [Block × 4] → 数字(10クラス)
                                 └─ 同じ ─┘
```

# はじめに

以下のコードを実行し、torch / torchvision が入っていて、MNIST とトークナイザがダウンロードできるかを確認する

```bash
uv run python check_env.py
```

# ディレクトリ構造

```
all_src/
├── W2/                      # 胴体(W6 から import して使う)
│   ├── block.py             # Block(Attention + FFN + 残差 + LayerNorm)
│   ├── layernorm.py
│   ├── data_utils.py        # load_data(文字単位トークナイザ stoi / itos を含む)
│   └── data/
│       └── input.txt        # tiny_shakespeare
└── W6/
    ├── README.md
    ├── check_env.py         # 環境確認
    ├── patch_embed.py       # 入口:画像 → パッチ → トークン
    ├── img_model.py         # ImgMini(パッチ埋め込み + W2 の Block + 分類ヘッド)
    ├── data_img.py          # MNIST の読み込みと get_batch
    ├── train_img.py         # 学習・評価・CSV 保存
    ├── plot.py              # 学習曲線の描画
    ├── tok_compare.py       # トークナイザ比較
    ├── data/                # MNIST(.gitignore で除外)
    │   └── MNIST/raw/
    └── temp/
        └── base/            # train_img.py base の出力
            ├── log_base.csv
            └── curve_base.png
```

# 実行

```bash
uv run python train_img.py base     # 学習。temp/base/log_base.csv に保存
uv run python plot.py base          # グラフ。temp/base/ に保存
uv run python tok_compare.py        # トークナイザ比較
```

---

## 1. 画像パッチ入力(MNIST)

### 構成

| ファイル | 役割 |
|---|---|
| `patch_embed.py` | 画像 `(B,1,28,28)` を 7×7 のパッチ16個に切り、`Linear(49→64)` でトークン `(B,16,64)` にする |
| `img_model.py` | `ImgMini`:パッチ埋め込み + 位置埋め込み + W2 の `Block`×4 + `ln_f` + 最後のパッチ `x[:, -1]` + `Linear(64→10)` |
| `data_img.py` | MNIST を読み、`float → /255 → unsqueeze(1)` で整え、`get_batch(split, batch_size, generator)` で取り出す |
| `train_img.py` | 学習ループ、評価(train/val 各20バッチ固定)、表の表示、CSV 保存 |
| `plot.py` | CSV から CE・正解率・gap のグラフを描く(W5 から流用、基準線を ln(10) に変更) |

- パラメータ数:204,170
- 設定:d_model=64、n_layer=4、patch_size=7、batch_size=256、lr=1e-3、AdamW、seed=0
- val はテスト用1万枚を使用(学習用6万枚とは配布元で分かれている)

### 結果(500ステップ)

| iter | train CE | train acc | val CE | val acc |
|---|---|---|---|---|
| 0 | 2.4703 | 11.0% | 2.4797 | 10.7% |
| 100 | 0.4369 | 87.5% | 0.4028 | 88.7% |
| 200 | 0.2344 | 92.5% | 0.2365 | 92.9% |
| 300 | 0.1605 | 95.2% | 0.1556 | 95.6% |
| 400 | 0.1691 | 94.7% | 0.1648 | 95.2% |
| 499 | 0.1187 | 96.5% | 0.1236 | 96.3% |

### 予想と結果

| | 予想 | 結果 |
|---|---|---|
| 学習前の loss(本物の MNIST) | 変わらない(2.3 前後) | 2.47。ln10 ≒ 2.30 に近い |
| 500ステップ後の val CE | 0.20 | 0.124 |
| 500ステップ後の val acc | 50% | 96.3% |

### 分かったこと

- 入口で `(B, T, d_model)` にそろえれば、W2 の胴体は1行も変えずに画像でも学習する
- 学習前の loss は ln(クラス数) + 0.16 付近。文字モデル(ln65 ≒ 4.17 に対し 4.33)と同じずれ方で、入力の種類によらない
- CE と正解率は独立に動かない。間違えた1枚の loss は −ln(0.5) ≒ 0.69 以上なので、正解率50%なら平均 CE は 0.35 以上になる。「CE 0.20 で正解率50%」は起こらない
- train と val の差は ±0.03 で、0 の周りを上下するだけ。過学習なし(500×256 ≒ 12.8万枚 = 約2周)
- gap のグラフは縦軸が自動で狭い範囲に合わせられ、大きく揺れて見えた。比較するときは縦軸を固定する
- 画像は画素値がそのまま `Linear` に入るので、`/255` で大きさをそろえる前処理が必要。文字は埋め込み表がこの役割を担っていた

---

## 2. トークナイザ比較

### 比べたもの

| トークナイザ | 方式 | 主な学習データ |
|---|---|---|
| char(W2) | 1文字 = 1トークン | tiny_shakespeare の65文字 |
| tiktoken(`o200k_base`) | byte-level BPE | 英語・コード中心 |
| llm-jp(`llm-jp-3-1.8b`) | Unigram(SentencePiece 系) | 日本語を多く含む |

### 比べた文

- ja:`吾輩は猫である。名前はまだ無い。`
- en:`To be, or not to be, that is the question.`
- code:`def forward(self, x):\n    return self.proj(x) + x`

### 結果:トークン数

| text | chars | bytes | char(W2) | tiktoken | llm-jp | missing | missing_text |
|---|---|---|---|---|---|---|---|
| ja | 16 | 48 | KeyError | 14 | 10 | 14 | 。,あ,い,だ,で,は,ま,る,前,名,吾,無,猫,輩 |
| en | 42 | 42 | 42 | 13 | 13 | 0 | |
| code | 49 | 49 | KeyError | 15 | 19 | 3 | (,),+ |

missing は W2 の語彙にない文字の種類数。

### 圧縮率(バイト数 ÷ トークン数)

| text | tiktoken | llm-jp |
|---|---|---|
| ja | 3.4 | 4.8 |
| en | 3.2 | 3.2 |
| code | 3.3 | 2.6 |

### 結果:区切り方

```
[ja] tiktoken(14): '吾' | '�' | '�' | 'は' | '猫' | 'で' | 'ある' | '。' | '名前' | 'は' | 'まだ' | '無' | 'い' | '。'
[ja] llmjp(10):    '▁' | '吾' | '輩' | 'は' | '猫' | 'である。' | '名前' | 'はまだ' | '無い' | '。'
[en] tiktoken(13): 'To' | ' be' | ',' | ' or' | ' not' | ' to' | ' be' | ',' | ' that' | ' is' | ' the' | ' question' | '.'
[en] llmjp(13):    '▁To' | '▁be' | ',' | '▁or' | '▁not' | '▁to' | '▁be' | ',' | '▁that' | '▁is' | '▁the' | '▁question' | '.'
[code] tiktoken(15): 'def' | ' forward' | '(self' | ',' | ' x' | '):\n' | '   ' | ' return' | ' self' | '.pro' | 'j' | '(x' | ')' | ' +' | ' x'
[code] llmjp(19):    '▁def' | '▁forward' | '(' | 'self' | ',' | '▁x' | '):' | '<0x0A>' | '▁▁▁' | '▁return' | '▁self' | '.' | 'pro' | 'j' | '(' | 'x' | ')' | '▁+' | '▁x'
```

### 予想と結果

| 予想 | 結果 |
|---|---|
| 日本語で一番トークン数が少ないのは llm-jp | 当たり(10 < 14) |
| コードで一番トークン数が少ないのは tiktoken | 当たり(15 < 19) |
| W2 の文字単位に日本語を入れると、学習がうまくいかない | 学習の前、トークンに変える段階で `KeyError` で止まる |
| llm-jp は「吾輩は猫である」を `吾輩 / は / 猫 / で / ある` で切る | `吾 / 輩 / は / 猫 / である。`。「吾輩」は分かれ、「である」は句点とくっついた |
| tiktoken は `    return` の空白4つを単独のトークンにしない | 空白3つが単独で1トークン、残り1つは `' return'` にくっついた |

### 分かったこと

- トークナイザには言語ごとの得意・不得意がある。日本語は llm-jp、コードは tiktoken が少ないトークンで表せ、英語は同じ
- 区切りは文法ではなく頻度で決まる。llm-jp の `'はまだ'` は助詞と副詞がくっついており、`'である。'` も文末でよく並ぶので1つになった。まれな「吾輩」は分かれた
- だから、得意な言語は学習データで決まる
- 文字単位の語彙は「学習データに出てきた文字」だけで決まる。止まるのは日本語だけではなく、シェイクスピアに出てこない `(` `)` `+` を含むコードでも止まる
- byte-level の方式は、知らない文字もバイトに分解できるので止まらない。ただし tiktoken の「輩」(UTF-8 で3バイト)は2つの断片に分かれ、どちらも単独では文字に戻せず `'�'` になった
- llm-jp は語彙にない改行を `'<0x0A>'`(バイトに分解して表す仕組み)で表した
- 日本語の1文字は UTF-8 で3バイト。文字数で割ると言語間で比べにくいので、圧縮率はバイト数で測る

### この比較で言えないこと

- 日本語で llm-jp が少なかった理由が、学習データの違いなのか、アルゴリズム(BPE / Unigram)の違いなのかは分けられない。分けるには、語彙数をそろえ、学習データとアルゴリズムを1つずつ変えて自分でトークナイザを学習させる必要がある
- 測ったのはトークナイザの効率で、モデル自体が日本語やコードをどれだけ上手に扱えるかは測っていない


## 参考文献

### 実装

- [画像データを学習・検証として扱う](https://qiita.com/taiga10969/items/24d85860ffcd724de9eb)
- [トークナイザーの実装](https://qiita.com/keishi_irisa/items/72a02be64d1ec34a706f)

### 比較したトークナイザ

| トークナイザ | URL |
|---|---|
| char(W2) | 自作(`all_src/W2/data_utils.py`)。元にしたもの:[karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture) |
| tiktoken(`o200k_base`) | [openai/tiktoken](https://github.com/openai/tiktoken) |
| llm-jp(`llm-jp-3-1.8b`) | モデル:[llm-jp/llm-jp-3-1.8b](https://huggingface.co/llm-jp/llm-jp-3-1.8b)<br>トークナイザ:[llm-jp/llm-jp-tokenizer](https://github.com/llm-jp/llm-jp-tokenizer) |

### 方式(BPE / Unigram)の理解

- [Byte Pair Encoding vs. Unigram Tokenization: A Deep Dive into Subword Models(Medium)](https://medium.com/@xiaxiami/byte-pair-encoding-vs-unigram-tokenization-a-deep-dive-into-subword-models-4963246e9a34)