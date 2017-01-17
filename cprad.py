"""
First created - 5th Dec 2016
File which alters the radius at which the cp is defined from the electrode probe posistion. It then works out the recall
for both groups and records them (this metric can be changed).

range of circular aperture radius's are used (thresholds)

input = pandas dataframe
"""

import pandas as pd
import cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics
from Functions import print_progress

datafile = raw_input("Pandas dataframe to open: ")
savefile = raw_input("Save name: ")
X = pd.read_hdf("%s.h5" % datafile)
cross_ref = X.pop('Distance 0')

recalls = list()
thresholds = range(9, 20, 1)
for t in thresholds:
    print "\n"
    y = cross_ref.apply(lambda x: 1.0 if x <= t else 0.0)
    print y.value_counts()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size= 0.25)
    dtree = RandomForestClassifier(n_estimators=15)
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    score = metrics.confusion_matrix(y_test, y_pred)
    print score
#     recalls.append(score)
#
# output = zip(*recalls)
# with open('%s.p' % savefile, 'wb') as f:
#     cPickle.dump(output, f)
