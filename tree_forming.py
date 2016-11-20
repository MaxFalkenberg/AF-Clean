# SingleSource_ECGdata_Itt1000_P60_df
import os
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sklearn.metrics as metrics
# from Functions import visualize_tree
from sklearn.cross_validation import cross_val_score
import cPickle

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf("%s.h5" % datafile)
del X['Distance']
del X['Crit Position']
del X['Probe Position']
y = X.pop('Target')
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

dtree = RandomForestClassifier(n_estimators=10)
scores = cross_val_score(dtree, X, y, cv=10, scoring='accuracy')

print scores
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
# visualize_tree(dtree, feature_names=X_train.columns)
print metrics.classification_report(y_test, y_pred)
print metrics.confusion_matrix(y_test, y_pred)
print metrics.mean_absolute_error(y_test, y_pred)

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
