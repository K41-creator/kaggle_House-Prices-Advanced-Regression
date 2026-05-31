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

'''
print(train.groupby("LandContour")["SalePrice"].mean()) 
print(train.groupby("Utilities")["SalePrice"].mean()) *
print(train.groupby("Neighborhood")["SalePrice"].mean()) *
print(train.groupby("Condition1")["SalePrice"].mean()) 
print(train.groupby("Condition2")["SalePrice"].mean()) *
print(train.groupby("BldgType")["SalePrice"].mean()) 
print(train.groupby("HouseStyle")["SalePrice"].mean()) (*) 
print(train.groupby("OverallQual")["SalePrice"].mean()) **
print(train.groupby("OverallCond")["SalePrice"].mean()) 
print(train.groupby("YearBuilt")["SalePrice"].mean()) *
print(train.groupby("YearRemodAd")["SalePrice"].mean()) //Nan (*)
print(train.groupby("Exterior1st")["SalePrice"].mean()) *
print(train.groupby("Exterior2nd")["SalePrice"].mean()) *
print(train.groupby("MasVnrType")["SalePrice"].mean()) *
print(train.groupby("Foundation")["SalePrice"].mean()) *
'''
##print(train.groupby("")["SalePrice"].mean())
##print(train.groupby("")["SalePrice"].mean())

'''
print(train.groupby("MasVnrArea")["SalePrice"].mean()) 
print(train.groupby("ExterQual")["SalePrice"].mean()) *
print(train.groupby("ExterCond")["SalePrice"].mean())
print(train.groupby("BsmtQual")["SalePrice"].mean()) *
print(train.groupby("BsmtCond")["SalePrice"].mean()) *
print(train.groupby("BsmtExposure")["SalePrice"].mean()) *
print(train.groupby("BsmtFinType1")["SalePrice"].mean()) *
print(train.groupby("GrLivArea")["SalePrice"].mean())
print(train.groupby("BsmtFullBath")["SalePrice"].mean())
print(train.groupby("BsmtHalfBath")["SalePrice"].mean())
print(train.groupby("FullBath")["SalePrice"].mean())
print(train.groupby("HalfBath")["SalePrice"].mean())
print(train.groupby("BedroomAbvGr")["SalePrice"].mean())
print(train.groupby("KitchenQual")["SalePrice"].mean()) *
print(train.groupby("TotRmsAbvGrd")["SalePrice"].mean()) *
print(train.groupby("GarageType")["SalePrice"].mean()) *
print(train.groupby("GarageYrBlt")["SalePrice"].mean()) *
print(train.groupby("GarageCars")["SalePrice"].mean()) *
print(train.groupby("GarageQual")["SalePrice"].mean()) *
print(train.groupby("WoodDeckSF")["SalePrice"].mean())
print(train.groupby("OpenPorchSF")["SalePrice"].mean())
print(train.groupby("EnclosedPorch")["SalePrice"].mean())
print(train.groupby("3SsnPorch")["SalePrice"].mean())
print(train.groupby("ScreenPorch")["SalePrice"].mean())
print(train.groupby("PoolArea")["SalePrice"].mean())

'''
"""
##qual の項目を使って評価する
corr = train.corr(numeric_only=True)

plt.figure(figsize=(12, 10))
sns.heatmap(corr)

plt.show()
"""
"""
OverallQual
YearBuilt
YearRemodAdd
MasVnrArea
TotalBsmtSF
1stFlrSF
GrLivArea
TotRmsAbvGrd
GarageCars
GarageArea
"""

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
    "GarageArea"
]

x = pd.get_dummies(train[features])

x_test = pd.get_dummies(test[features])

y = train["SalePrice"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(
    x,
    y
    )

predictions = model.predict(x_test)

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)