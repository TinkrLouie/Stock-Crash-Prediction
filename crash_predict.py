from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

df = pd.read_csv('ncskew.csv')

n_feats = np.size(df, 1) - 1

X = df.iloc[:, :n_feats].values
y = df.iloc[:, n_feats:].values
train, test = train_test_split(df, train_size=0.8, random_state=42)
X_train = train.iloc[:, :n_feats].values
y_train = train.iloc[:, n_feats:].values
X_test = test.iloc[:, :n_feats].values
y_test = test.iloc[:, n_feats:].values

trans = PolynomialFeatures(degree=2)

X_train = trans.fit_transform(X_train)
X_test = trans.fit_transform(X_test)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_train, y_train)
train_accuracy = lin_reg_poly.score(X_train, y_train)
test_accuracy = lin_reg_poly.score(X_test, y_test)

print('Train score: ', train_accuracy)
print('Test score: ', test_accuracy)