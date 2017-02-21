"""
File which finds the robustness of the features.
"""

import os
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from Functions import print_progress
import cPickle

while True:
    input_list = ['c', 'r']
    RFtype = raw_input("Classifier or Regressor (c/r): ")
    if RFtype in input_list:
        break

model_number = int(raw_input("Number of models: "))

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf("%s.h5" % datafile)
if RFtype == 'c':
    y = X.pop('Target 0')
    y = y.astype(int)
if RFtype == 'r':
    y = X.pop('Distance 0')

rows = model_number
# For finding feature robustness for multi-electrode
# y = X.pop('Multi Target 0')
# y = y.astype(int)
#
# rows = 15
robustness_datagrid = np.zeros((rows, len(X.columns)))

pp = 0
print_progress(pp, rows, prefix='Progress:', suffix='Complete', bar_length=50)
for row in range(rows):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    if RFtype == 'c':
        forest = RandomForestClassifier(n_estimators=15)
    if RFtype == 'r':
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
