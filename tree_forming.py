import os
import pandas as pd
from Functions import feature_prune
from sklearn.cross_validation import train_test_split
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import cPickle

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf(os.path.join('Data', "%s.h5" % datafile))
feature_prune(X, ['Distance', 'Crit Position', 'Probe Position',
                  'Unit Vector X', 'Unit Vector Y', 'Theta', 'Sample Length'])
feature_prune(X, ['Largest FT Mag %s' % x for x in range(1, 10)])
feature_prune(X, ['Largest FT Freq %s' % x for x in range(1, 10)])
y = X.pop('Target')
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

dtree = RandomForestClassifier(n_estimators=15)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.mean_absolute_error(y_test, y_pred))

print '\n'
save_deci = raw_input("Save model (y/n): ")
if save_deci == 'y':
    modelname = raw_input("filename: ")
    MY_dIR = os.path.realpath(os.path.dirname(__file__))
    PICKLE_DIR = os.path.join(MY_dIR, 'ML_models')
    fname = os.path.join(PICKLE_DIR, '%s.p' % modelname)
    with open(fname, 'wb') as f:
        cPickle.dump(dtree, f)
else:
    pass
