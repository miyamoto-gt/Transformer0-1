### 参考文献
MSE vs CE
https://qiita.com/Seine_A_Shintani/items/329867cbd32a01da452b
adamw
https://zenn.dev/bilzard/articles/adamw-demistified
損失関数における最適化アルゴリズム
https://zenn.dev/mutex_inc/articles/2948e2c5934171


### compare.pngのy軸における意味
*   val ce 
    val の各位置で計算した −log p の平均。p は、モデルが正解の文字に置いた確率のこと。
    見たことのない文章を1文字ずつ読ませたとき、モデルが次の文字に平均してどれだけ驚いたか
* val top-1 accuracy
    1位に予測した文字が正解と一致した位置の割合。⭕️か❌か
* lm_head grad norm
    出力層(lm_head)の重みの勾配の大きさ
