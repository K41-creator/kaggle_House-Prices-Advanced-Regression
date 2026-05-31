# kaggle_House-Prices-Advanced-Regression

Kaggle の「House Prices - Advanced Regression Techniques」に取り組み、住宅価格予測モデルを構築しました。

## Competition Overview

住宅の特徴量から販売価格（SalePrice）を予測する回帰問題です。

- Training Samples: 1460
- Test Samples: 1459
- Features: 79
- Evaluation Metric: RMSE（対数価格）

本コンペでは、データ前処理・特徴量エンジニアリング・モデル選択を繰り返しながらスコア改善を行いました。

---

## Goal

住宅の情報から販売価格を予測するモデルを構築し、Kaggle の Public Leaderboard スコア向上を目指しました。

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost

---

## Project Workflow

### 1. Data Exploration

まずデータ構造を確認しました。

```python
train.head()
train.info()
train.isnull().sum()
```

- 各特徴量の意味を確認
- 欠損値の有無を確認
- 価格との関係が強そうな特徴量を調査

---

### 2. Train/Test Integration

カテゴリ変数の不一致を防ぐため、学習データとテストデータを結合して前処理を行いました。

```python
all_data = pd.concat(
    [train.drop("SalePrice", axis=1), test],
    axis=0
)
```

---

### 3. Missing Value Handling

欠損値を含む特徴量を調査し、一部は情報として活用しました。

例：

```python
all_data["GarageType"] = all_data["GarageType"].fillna("None")
```

GarageType の欠損は「ガレージが存在しない」ことを意味するため、欠損値として削除せずに利用しました。

---

### 4. One-Hot Encoding

カテゴリ変数を数値化するために One-Hot Encoding を実施しました。

```python
all_data = pd.get_dummies(all_data)
```

---

### 5. Target Transformation

コンペの評価指標が対数 RMSE であるため、目的変数に対数変換を適用しました。

```python
y = np.log1p(train["SalePrice"])
```

予測時は元のスケールへ戻しています。

```python
predictions = np.expm1(predictions)
```

---

### 6. Feature Engineering

価格との関係を考えながら新しい特徴量を作成しました。

#### TotalSF

住宅全体の床面積

```python
all_data["TotalSF"] = (
    all_data["TotalBsmtSF"]
    + all_data["1stFlrSF"]
    + all_data["2ndFlrSF"]
)
```

#### HouseAge

住宅の築年数

```python
all_data["HouseAge"] = (
    all_data["YrSold"]
    - all_data["YearBuilt"]
)
```

---

### 7. Model Selection

#### Random Forest

最初のベースラインモデルとして使用。

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

---

#### XGBoost

最終モデルとして採用。

```python
XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

採用理由：

- 表形式データとの相性が良い
- 過学習を抑えやすい
- RandomForest より高い精度を実現できた

---

### 8. Cross Validation

モデル評価には 5-Fold Cross Validation を使用しました。

```python
scores = cross_val_score(
    model,
    x,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

print(-scores.mean())
```

提出前に性能を確認することで、効率的な改善サイクルを実現しました。

---

## Score Improvement History

| Improvement Step | Public Score |
|------------------|-------------:|
| RandomForest Baseline | 0.16188 |
| Data Preprocessing Improvement | 0.14738 |
| Log Transformation | 0.14687 |
| XGBoost Introduction | 0.13836 |
| Feature Engineering | 0.12940 |
| Additional Tuning | **0.12917** |

---

## What I Learned

このプロジェクトを通じて以下を学びました。

### Data Preprocessing

- 欠損値処理
- One-Hot Encoding
- Train/Test 結合による前処理

### Feature Engineering

- ドメイン知識を用いた特徴量作成
- 特徴量がモデル性能に与える影響

### Machine Learning

- 回帰分析
- Random Forest
- Gradient Boosting
- XGBoost

### Model Evaluation

- Cross Validation
- RMSE
- Public Leaderboard と過学習の関係

### Kaggle Workflow

```text
仮説を立てる
↓
実装する
↓
クロスバリデーションで評価
↓
提出する
↓
結果を分析する
↓
改善する
```

---

## Final Result

**Public Score: 0.12917**

RandomForest によるベースラインモデルからスタートし、前処理・特徴量エンジニアリング・XGBoost の導入を通じてスコアを大幅に改善することができました。

---

## References

- Kaggle House Prices Competition
- Scikit-Learn Documentation
- XGBoost Documentation