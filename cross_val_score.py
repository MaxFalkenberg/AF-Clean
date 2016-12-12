"""
This doesn't do anything as of yet.
"""


import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

datafile = raw_input("Pandas dataframe to open: ")
forest = RandomForestClassifier(n_estimators=3)
X = pd.read_hdf("%s.h5" % datafile)
y = X.pop('Target')
y = y.astype(int)

scores = cross_val_score(forest, X, y, cv=2, scoring='recall', )

print scores
# print np.mean(scores)
# print np.std(scores, ddof=1)
