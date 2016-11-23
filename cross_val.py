# SingleSource_ECGdata_Itt1000_P60_df
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from Functions import print_progress, feature_prune
import os
import cPickle

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf(os.path.join('Data', "%s.h5" % datafile))
feature_prune(X, ['Distance', 'Crit Position', 'Probe Position',
                  'Unit Vector X', 'Unit Vector Y', 'Theta', 'Sample Length'])
feature_prune(X, ['Largest FT Mag %s' % x for x in range(1, 10)])
feature_prune(X, ['Largest FT Freq %s' % x for x in range(1, 10)])
y = X.pop('Target')
y = y.astype(int)

mean_scores_l = list()
errors = list()
progress = 0
n_tree_range = range(1, 25)
print_progress(progress, len(n_tree_range), prefix='Progress', suffix='Complete', bar_length=50)
for i in n_tree_range:
    dtree = RandomForestClassifier(n_estimators=i)
    scores = cross_val_score(dtree, X, y, cv=10, scoring='accuracy')
    mean_scores_l.append(np.mean(scores))
    errors.append(np.std(scores))
    progress += 1
    print_progress(progress, len(n_tree_range), prefix='Progress', suffix='Complete', bar_length=50)

print mean_scores_l
print errors
data = zip(mean_scores_l, errors)
dataname = raw_input("filename: ")
MY_dIR = os.path.realpath(os.path.dirname(__file__))
PICKLE_DIR = os.path.join(MY_dIR, 'ML_models')
fname = os.path.join(PICKLE_DIR, '%s.p' % dataname)
with open(fname, 'wb') as f:
    cPickle.dump(data, f)
