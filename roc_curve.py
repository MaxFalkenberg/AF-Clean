"""
Find the roc curve for the decision tree. (Needs to be updated as the features are pruned before they are entered into
files now.)
"""

import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics
import seaborn as sns
from Functions import feature_prune

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf("%s.h5" % datafile)
feature_prune(X, ['Distance', 'Crit Position', 'Probe Position',
                  'Sample Length'])
# feature_prune(X, ['Distance', 'Crit Position', 'Probe Position',
#                   'Unit Vector X', 'Unit Vector Y', 'Theta', 'Sample Length'])
# feature_prune(X, ['Crit Position 0', 'Crit Position 1', 'Probe Position', 'Distance 0', 'Distance 1',
#                   'Unit Vector X 0', 'Unit Vector X 1', 'Unit Vector Y 0',
#                   'Unit Vector Y 1', 'Theta 0', 'Theta 1', 'Target 0', 'Target 1',
#                   'Nearest Crit Position', 'True Distance'])
feature_prune(X, ['Largest FT Mag %s' % x for x in range(1, 10)])
feature_prune(X, ['Largest FT Freq %s' % x for x in range(1, 10)])
y = X.pop('Target')
y = y.astype(int)

# should put this somewhere where it opens a ML file
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
dtree = RandomForestClassifier(n_estimators=15, random_state=1)
dtree.fit(X_train, y_train)
prob_pred = dtree.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, prob_pred)
roc_auc = metrics.auc(fpr, tpr)
print roc_auc
sns.plt.plot(fpr, tpr)
sns.plt.xlabel("FPR")
sns.plt.ylabel("TPR")
sns.plt.show()
