import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from sklearn import metrics

df = pd.read_csv('crash.csv')

n_feats = np.size(df, 1) - 1

X = df.iloc[:, :n_feats].values
y = df.iloc[:, n_feats:].values
train, test = train_test_split(df, train_size=0.8, random_state=42)
X_train = train.iloc[:, :n_feats].values
y_train = train.iloc[:, n_feats:].values
X_test = test.iloc[:, :n_feats].values
y_test = test.iloc[:, n_feats:].values

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = keras.models.Sequential()

#1st layer
ann.add(keras.layers.Dense(units=6, activation="relu"))
#2nd layer
ann.add(keras.layers.Dense(units=6, activation="relu"))
#Adding Output Layer
ann.add(keras.layers.Dense(units=1, activation="sigmoid"))
#Compiling ANN
ann.compile(optimizer="Adam", loss=keras.losses.CategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy()])

ann.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

ann.save("ANN_crash.h5")

#predictions = list(ann.predict(X_test))
#score = metrics.accuracy_score(y_test, predictions)
#
#print('Model Accuracy: {0:f}'.format(score))