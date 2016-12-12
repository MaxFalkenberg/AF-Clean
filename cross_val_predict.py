"""
File which uses cross_val_predict to find metric scores.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict
import sklearn.metrics as metrics
from Functions import print_progress
import cPickle

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf("%s.h5" % datafile)
y = X.pop('Target')
y = y.astype(int)

progress = 0
# range(1, 25)
n_tree_range = [3, 4]
print_progress(progress, len(n_tree_range), prefix='Progress', suffix='Complete', bar_length=50)
for i in n_tree_range:
    forest = RandomForestClassifier(n_estimators=i)
    y_pred = cross_val_predict(forest, X, y, cv=2)
    print metrics.recall_score(y, y_pred, average=None)
    progress += 1
    print_progress(progress, len(n_tree_range), prefix='Progress', suffix='Complete', bar_length=50)

# data = zip(mean_scores_l, errors)
# dataname = raw_input("filename: ")
# with open('%s.p' % dataname, 'wb') as f:
#     cPickle.dump(data, f)
