from keras.models import load_model
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ann = load_model('ANN_crash.h5')

df = pd.read_csv('crash.csv')

n_feats = np.size(df, 1) - 1

X = df.iloc[:, :n_feats].values
y = df.iloc[:, n_feats:].values
train, test = train_test_split(df, train_size=0.8, random_state=42)
X_test = test.iloc[:, :n_feats].values
y_test = test.iloc[:, n_feats:].values

predictions = list(ann.predict(X_test))
score = metrics.accuracy_score(y_test, predictions)

print('Model Accuracy: {0:f}'.format(score))