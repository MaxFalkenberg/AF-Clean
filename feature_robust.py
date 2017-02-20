"""
File which finds the robustness of the features.
"""

import os
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from Functions import print_progress
import cPickle
from Functions import feature_prune

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf("%s.h5" % datafile)
# y = X.pop('Multi Target 0')
# y = y.astype(int)

y = X.pop('Vector Y 0')
feature_prune(X, [ 'Target 0', 'Vector X 0', 'Multi Target 0', 'Nu', 'Theta 0', 'Distance 0'])

rows = 5
robustness_datagrid = np.zeros((rows, len(X.columns)))

pp = 0
print_progress(pp, rows, prefix='Progress:', suffix='Complete', bar_length=50)
for row in range(rows):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    forest = RandomForestRegressor(n_estimators=15)
    forest.fit(X_train, y_train)
    robustness_datagrid[row] = forest.feature_importances_
    pp += 1
    print_progress(pp, rows, prefix='Progress:', suffix='Complete', bar_length=50)

mean_importance = np.mean(robustness_datagrid, axis=0)

sorted_mean_importance = np.sort(mean_importance)
sorted_std_importance = np.sort(np.std(robustness_datagrid, axis=0, ddof=1))
indicies = np.argsort(mean_importance)[::1]
feature_names_sorted = [X.columns[ind] for ind in indicies]

with open('%s_robust.p' % datafile, 'wb') as f:
    cPickle.dump((sorted_mean_importance, sorted_std_importance, feature_names_sorted), f)
