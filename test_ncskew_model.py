from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

ann = load_model('ANN_ncskew.h5', compile=False)

df = pd.read_csv('ncskew.csv')

batch_size = 64
n_feats = np.size(df, 1) - 1

train, test = train_test_split(df, train_size=0.8, random_state=42)
X_train = train.iloc[:, :n_feats]
y_train = train.iloc[:, n_feats:]
X_test = test.iloc[:, :n_feats]
y_true = test.iloc[:, n_feats:]
y_true.rename(columns={"ncskew_new_plus": "y_true"}, inplace=True)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Prediction and Score
predictions = ann.predict(X_test)


def flatten(xss):
    return [x for xs in xss for x in xs]


predictions = flatten(predictions)

#Graphs
y_pred = pd.DataFrame({'y_pred': predictions})
print("y_pred statistics:")
print(y_pred.describe())
print("y_true statistics:")
print(y_true.describe())
df_result = pd.concat([y_pred.reset_index(drop=True), y_true.reset_index(drop=True)], axis=1)

resid = sns.residplot(data=df_result, x="y_true", y="y_pred")

g = sns.lmplot(x='y_true', y='y_pred', data=df_result)
g.fig.suptitle('True Vs Pred', y=1.02)
g.set_axis_labels('y_true', 'y_pred')
plt.show()