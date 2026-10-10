# W4: 損失関数と出口を差し替える

## ディレクトリ構造

```
all_src/
  W2/                      ← 読むだけ。編集しない
    model.py               GPTMini
    data_utils.py          load_data
    ckpt.pt                W2 の学習済み重み(sample_temp.py の既定)
    data/input.txt         tiny_shakespeare
  W4/
    base_train.py          共通処理(学習・評価・保存)
    exp_ce.py              実験1: CE で学習
    exp_mse.py             実験1: MSE で学習
    exp_mse_lr.py          実験1: MSE + 学習率×10
    plot_log.py            学習ログをグラフ化
    eval_ckpt.py           保存した重みを複数の物差しで採点
    sample_temp.py         実験2: 温度 T ごとの生成比較
    docs/                  実験ごとのまとめ
      eval_ckpt.md         実験1(CE vs MSE)の結果
      temperature.md       実験2(温度 T)の結果
      notes.md             分かったこと・残った問い
    temp/                  出力先(ckpt / log / sample / compare.png)
```

## 各ファイルの説明

| ファイル | 役割 | 出力(`temp/`) |
|---|---|---|
| `base_train.py` | 設定・データ読み込み・`run(loss_fn, tag, lr)` を持つ。import しても学習は走らない | — |
| `exp_ce.py` | `loss_fn` を交差エントロピーにして `run()` | `log_ce.csv` `ckpt_ce.pt` `sample_ce.txt` |
| `exp_mse.py` | `loss_fn` を softmax 後の確率と one-hot の二乗誤差にして `run()` | `log_mse.csv` `ckpt_mse.pt` `sample_mse.txt` |
| `exp_mse_lr.py` | `exp_mse.py` の `loss_fn` を流用し、lr を 10 倍にして `run()` | `log_mse_lr*10.csv` ほか |
| `plot_log.py` | `log_*.csv` を読み、val CE / val 正解率 / lm_head 勾配ノルムを3枚並べて描く | `compare.png` |
| `eval_ckpt.py` | `ckpt_*.pt` を読み、val 全体で val CE・val MSE・正解率・p 中央値・p<0.01・p<0.001 を表示。ランダム性なし | 標準出力のみ |
| `sample_temp.py` | 重みを読み、T = 0.5 / 0.8 / 1.0 / 1.5 / 2.0 で生成。エントロピー・単語の重複率・実在単語率を表示。学習はしない | `sample_<tag>_T<T>.txt` |

実験ファイルは `loss_fn` を定義して `run()` を呼ぶだけ。変えるのは損失関数(と lr)の1か所のみ。

### 共通の設定(`base_train.py`)

| 変数 | 値 |
|---|---|
| `d_model` / `n_layer` | 64 / 4 |
| `block_size` / `batch_size` | 32 / 32 |
| `lr` | 1e-3(AdamW) |
| `max_iters` / `eval_interval` / `eval_iters` | 3000 / 300 / 50 |
| `seed` | 0 |

### compare.png の y 軸

- **val CE**: val の各位置で計算した −log p の平均。p は正解の文字に置いた確率
- **val top-1 accuracy**: 1位に予測した文字が正解と一致した位置の割合
- **lm_head grad norm**: 出力層(`lm_head`)の重みの勾配の大きさ(対数軸)

## 実行

```bash
cd all_src/W4
uv run python exp_ce.py                        # 実験1: CE
uv run python exp_mse.py                       # 実験1: MSE
uv run python exp_mse_lr.py                    # 実験1: MSE + lr×10
uv run python plot_log.py                      # ログをグラフ化
uv run python eval_ckpt.py                     # 全 ckpt を採点(引数で tag 指定可: ce mse)
uv run python sample_temp.py                   # 実験2: W2 の重みで温度比較
uv run python sample_temp.py --ckpt ckpt_ce.pt # 実験1 の重みで温度比較
```

結果の詳細は `docs/` を参照。
