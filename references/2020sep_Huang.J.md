# Keeping Dataset Biases out of the Simulation

## 研究概要
- シミュレータ用の評価値行列を構築する際に1. ポジティブバイアス（高い評価を持っているアイテムほど出現する）, 2. 人気度バイアス（出現回数の多いアイテムほど高い評価を持つ）といった二つのバイアスを修正して用いる方法を提案
- シミュレータの方策に対するバイアスの効果を評価する指標の提案


## デバイアスシミュレータ

- ユーザ-アイテム評価値行列とユーザーの選択モデルを使用
- 評価値行列を求める前にバイアスを修正するIBMS(Intermidiate Bias Mitigation Step)を挿入
- 本研究では傾向スコアとcomplete-case analysisを使用

<img src=imgs/simulator_framework.png width=50%><img src=imgs/ibms_ex.png width=50%>

### 通常のロス関数（ナイーブなロス）
$Y:$正解の評価値, $\hat{Y}: $予測評価値, $o_{u,i}:$ユーザがアイテムを評価したか, $\delta$ MSEやMAE

$$
\mathcal{L}_{Naive} = \frac{1}{|\{(u, i):o_{u,i}=1\}|}\sum_{(u, i):o_{u, i}=1}\delta_{u, i}(Y, \hat{Y})
$$

不偏推定量でない

### IPSロス
- 出現確率で重み付けした損失
- $P(o_{u, i}=1)$はナイーブベイズ最尤推定かユーザとアイテムに対するロジスティック回帰で求める

$$
\mathcal{L}_{IPS} = \frac{1}{N・M}\sum_{(u, i):o_{u, i}=1}\frac{\delta_{u, i}(Y, \hat{Y})}{P(o_{u, i}=1)}
$$






