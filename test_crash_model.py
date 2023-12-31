from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

ann = load_model('ANN_crash.h5')

df = pd.read_csv('crash.csv')

n_feats = np.size(df, 1) - 1

train, test = train_test_split(df, train_size=0.8, random_state=42)
X_test = test.iloc[:, :n_feats]
y_test = test.iloc[:, n_feats:]

predictions = ann.predict(X_test)


def flatten(xss):
    return [x for xs in xss for x in xs]


predictions = flatten(predictions)

score = accuracy_score(y_test, predictions)

print('Model Accuracy: {0:f}'.format(score))

print(classification_report(y_test, predictions))

#result = permutation_importance(
#    ann, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2, scoring="accuracy"
#)
#importances = pd.Series(result.importances_mean, index=list(X_test.columns.values))
#fig, ax = plt.subplots()
#importances.plot.bar(yerr=result.importances_std, ax=ax)
#ax.set_title("Feature importances using permutation on full model")
#ax.set_ylabel("Mean accuracy decrease")
#fig.tight_layout()

cf_matrix = confusion_matrix(y_test, predictions)
cf_matrix_plot = sns.heatmap(cf_matrix, annot=True, fmt=".1f")
plt.show()