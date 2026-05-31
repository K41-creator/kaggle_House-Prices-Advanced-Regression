import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor


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
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "GrLivArea",
    "TotRmsAbvGrd",
    "GarageCars",
    "GarageArea",
    "OverallQual",
    "ExterQual",
    "BsmtQual",
    "KitchenQual",
    "GarageQual"
]

y = np.log1p(train["SalePrice"])

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

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