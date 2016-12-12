"""
Loads a saved decision tree model and applys test data to it to get feature importance. (feature_robust.py is better)
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import seaborn as sns
import matplotlib.patches as mpatches

treefile = raw_input("ML file to load: ")
dtree = joblib.load("%s.p" % treefile)
datafile = raw_input("Pandas dataframe to open: ")

# Feature pruning and popping for the train test split
X = pd.read_hdf("%s.h5" % datafile)
y = X.pop('Target')
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

feature_imp = np.sort(dtree.feature_importances_)[::1]
indicies = np.argsort(dtree.feature_importances_)[::1]
feature_names_sorted = [X_train.columns[ind] for ind in indicies]

fig = sns.plt.figure()
colors = ['orange' if feat > np.mean(feature_imp) else 'blue' for feat in feature_imp]
sns.set(style="white")
sns.barplot(feature_imp, feature_names_sorted, palette=colors)
sns.plt.title("Feature Importance for %s" % treefile, fontsize=16)
sns.plt.xlabel("Feature Importance (Gini)", fontsize=14)
sns.despine(left=True, offset=10)
orange_patch = mpatches.Patch(color='orange', label='Above Mean')
blue_path = mpatches.Patch(color='blue', label='Below Mean')
sns.plt.legend(handles=[blue_path, orange_patch], prop={'size': 15})
sns.plt.show()
sns.plt.close()
