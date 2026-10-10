# W4 実験2:温度 T による生成の比較

## 条件

- 重み:W2 の ckpt.pt(学習はしない)
- 生成:改行から開始して 1000 文字、seed=0(どの T でも同じ乱数列)
- 変えたもの:softmax(logits / T) の T のみ

## 結果(sample_temp.py)

| T | 平均エントロピー(bit) | 実質の候補数(2^H) | 単語の重複率 | 実在単語率 |
|---|---|---|---|---|
| 0.5 | 1.313 | 約 2.5 | 0.527 | 0.896 |
| 0.8 | 1.925 | 約 3.8 | 0.374 | 0.841 |
| 1.0 | 2.309 | 約 5.0 | 0.225 | 0.653 |
| 1.5 | 3.352 | 約 10.2 | 0.088 | 0.372 |
| 2.0 | 4.065 | 約 16.7 | 0.061 | 0.197 |

- エントロピーの上限は log2(65) ≈ 6.02 bit
- 単語の重複率 = 1 − 異なり語数 / 総語数
- 実在単語率 = 生成した単語のうち train に出てくる単語の割合(短い単語は偶然一致しやすく、やや甘めに出る)

## サンプル(冒頭)

T=0.5

```
That make the prester our and hange in the strenge
That you than her shall the see thinks of his stants, and away,
That they are that we mother of well.
```

T=2.0

```
thyremhmadk'swes:? fildHad neamany,
Nding
let, grecourewy his tweetre hath Doig Oureliess. I him;sleat'ory? raius,
```

## 読み取れること

- T を上げると候補が約 2.5 → 約 16.7 文字に広がり、重複率と実在単語率がそろって下がった
- 重みは 5 条件すべてで同じ。変わったのは確率の配分だけ
- T=0.5 は読めるが `thou thou thou` などの繰り返しが目立つ。T=2.0 は `$`・`Q`・`&` など train にほぼ出ない文字まで選ばれた
- T=0.5 でも `tescrands`・`soffather` のような存在しない単語が混ざる。これはモデル自体の限界で、T では消せない