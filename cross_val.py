# SingleSource_ECGdata_Itt1000_P60_df
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from Functions import print_progress
import cPickle

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf("%s.h5" % datafile)
del X['Distance']
del X['Crit Position']
del X['Probe Position']
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
with open('cross-validation(RFC1).p', 'wb') as f:
    cPickle.dump(data, f)
