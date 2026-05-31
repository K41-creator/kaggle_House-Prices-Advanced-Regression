import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


pd.set_option('display.max_columns', None)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
##print(train.head())
##print(train.info)
##print(train.groupby("MSSubClass")["SalePrice"].mean())
##print(train.groupby("MSZoning")["SalePrice"].mean())

##testとtrainのデータを結合する

all_data = pd.concat(
    [train.drop("SalePrice",axis=1),test],
    axis=0
)
train_rows = len(train)




##欠損データの有効化
all_data["GarageType"] = all_data["GarageType"].fillna("None")








##特徴量エンジニアリング
all_data["TotalSF"] = (
    all_data["TotalBsmtSF"]
    + all_data["1stFlrSF"]
    + all_data["2ndFlrSF"]
)

all_data["HouseAge"] = (
    all_data["YrSold"]
    - all_data["YearBuilt"]
)








all_data = pd.get_dummies(all_data)


pd.set_option('display.max_columns', None)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

x = all_data.iloc[:len(train)]
x_test = all_data.iloc[len(train):]



##missing = train.isnull().sum()
##print(missing[missing != 0])




features = [
    "OverallQual",
    "HouseAge",
    "YearRemodAdd",
    "MasVnrArea",
    "TotalSF",
    "GrLivArea",
    "TotRmsAbvGrd",
    "GarageCars",
    "GarageType",
    "GarageArea",
    "OverallQual",
    "ExterQual",
    "BsmtQual",
    "KitchenQual",
    "GarageQual"
]

y = np.log1p(train["SalePrice"])
"""
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
"""

model.fit(
    x,
    y
    )

predictions = model.predict(x_test)
predictions = np.expm1(predictions)

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)





##検証

scores = cross_val_score(
    model,
    x,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

print(-scores.mean())