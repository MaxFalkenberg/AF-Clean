import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
from Functions import feature_prune
import os

treefile = raw_input("ML file to load: ")
dtree = joblib.load(os.path.join('ML_models', "%s.p" % treefile))
datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf(os.path.join('Data', "%s.h5" % datafile))
feature_prune(X, ['Distance', 'Crit Position', 'Probe Position',
                  'Unit Vector X', 'Unit Vector Y', 'Theta', 'Sample Length'])
feature_prune(X, ['Largest FT Mag %s' % x for x in range(1, 10)])
feature_prune(X, ['Largest FT Freq %s' % x for x in range(1, 10)])
y = X.pop('Target')
y = y.astype(int)

scores = cross_val_score(dtree, X, y, cv=5, scoring='accuracy')

print scores
print np.mean(scores)
