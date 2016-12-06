import os
import pandas as pd
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics
import cPickle

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf(os.path.join('Dataframes', "%s.h5" % datafile))
model_choice = raw_input("Regressor or Classifier (r\c): ")
save_deci = raw_input("Save model (y/n): ")
modelname = None
if save_deci == 'y':
    modelname = raw_input("filename: ")
dtree = None

if model_choice == 'r':
    from sklearn.ensemble import RandomForestRegressor
    y = X.pop('Distance')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    dtree = RandomForestRegressor(n_estimators=15)
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    print(metrics.mean_absolute_error(y_test, y_pred))

if model_choice == 'c':
    from sklearn.ensemble import RandomForestClassifier
    y = X.pop('Target')
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    dtree = RandomForestClassifier(n_estimators=15)
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

print '\n'
if save_deci == 'y':
    MY_dIR = os.path.realpath(os.path.dirname(__file__))
    PICKLE_DIR = os.path.join(MY_dIR, 'ML_models')
    fname = os.path.join(PICKLE_DIR, '%s.p' % modelname)
    with open(fname, 'wb') as f:
        cPickle.dump(dtree, f)
else:
    pass
