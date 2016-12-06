"""
First created - 5th Dec 2016
File which alters the radius at which the cp is defined from the electrode probe posistion. It then works out the recall
for both groups and records them (this metric can be changed).
"""

import pandas as pd
import os
import cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics
from Functions import print_progress

datafile = raw_input("Pandas dataframe to open: ")
savefile = raw_input("Save name: ")
X = pd.read_hdf(os.path.join('Dataframes', "%s.h5" % datafile))
cross_ref = X.pop('Distance')

recalls = list()
thresholds = range(5, 25, 1)
progress = 0
print_progress(progress, len(thresholds), prefix='Progress', suffix='Complete', bar_length=50)
for t in thresholds:
    y = cross_ref.apply(lambda x: 1.0 if x <= t else 0.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    dtree = RandomForestClassifier(n_estimators=15)
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    score = metrics.recall_score(y_test, y_pred, average=None)
    recalls.append(score)
    progress += 1
    print_progress(progress, len(thresholds), prefix='Progress', suffix='Complete', bar_length=50)

output = zip(*recalls)
MY_dIR = os.path.realpath(os.path.dirname(__file__))
PICKLE_DIR = os.path.join(MY_dIR, 'Data')
fname = os.path.join(PICKLE_DIR, '%s.p' % savefile)
with open(fname, 'wb') as f:
    cPickle.dump(output, f)
