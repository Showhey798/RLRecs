# Keeping Dataset Biases out of the Simulation

## 研究概要
- シミュレータ用の評価値行列を構築する際に1. ポジティブバイアス（高い評価を持っているアイテムほど出現する）, 2. 人気度バイアス（出現回数の多いアイテムほど高い評価を持つ）といった二つのバイアスを修正して用いる方法を提案
- シミュレータの方策に対するバイアスの効果を評価する指標の提案


## デバイアスシミュレータ

- ユーザ-アイテム評価値行列とユーザーの選択モデルを使用
- 評価値行列を求める前にバイアスを修正するIBMS(Intermidiate Bias Mitigation Step)を挿入
- 本研究では傾向スコアとcomplete-case analysisを使用
- 推薦システムにより収集されたデータはMissing Not At Randam(MNAR)なデータであるためバイアスが生じる

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



## バイアスが与えるシミュレータへの影響の評価方法

- 従来は生成したシミュレータを実際に取られたテストデータを用いて行われている
  - 　そのシミュレータを用いて学習した方策の性能を考えていない
- スパースなMissing Completely At Random(MCAR)な評価値を用いて評価する方法の提案

```
手順
1. MNARなデータに対してIBMSを搭載したシミュレータで方策を学習（debiased policy）
2. IBMSを使用しないしていないMNARデータ上のシミュレータを用いて方策を学習(biased policy)
3. MCARデータ上のシミュレータを作成
4. 3.のシミュレータ上でbiased policyとdebiased policyを実行
```

一方で、MCARなデータは収集が難しいため、スパースになりやすい

解決策
1. 評価時はMCARデータで欠損のない部分のみで評価を行う
   - 実際のMCARデータに基づいて評価が可能
   - 実際の推薦システムの挙動を評価することが難しい（評価を持っていないアイテムを推薦した場合、判定ができない） 
2. 評価値予測モデルを用いて欠損部分を補完
   - 全データで評価が**一応**可能
   - 予測値であるため、実際の好みと異なる可能性がある


## 提案法(Simulator for Offline LeArning and evaluation, SOFA)

1. MF-IPSを用いてdebiased user-item rating matrixを作成
2. ユーザ選択モデルを作成
    - Feedback Simulation : 評価ちが3以上をポジティブな嗜好と判定($f_t = 1$ otherwise $f_t=0$)
    - State trainsition : 過去の履歴$\{i_1, ..., i_t\}$とそのフィードバック$\{f_1,..., f_t\}$を連結したもの
    - Reward generation : クリックを1,スキップを-2として学習(予備実験でうまく行ったやつらしい)


## 実験

データセット
学習データがMNARでテストデータがMCARであるものを選択
- Yahoo!R3 dataset :ナイーブベイズで$P(o_{u,i}=1)$を学習
- Coat dataset : 生息かロジスティック回帰で$P(o_{u,i}=1)$を学習
- 人口データ（バイアスを強くした場合が見たいため）

評価方法
- 方策: 累積報酬和（10回中何回クリックされたか）
- シミュレータ：MSE, MAE, ACC, Click-ACC

