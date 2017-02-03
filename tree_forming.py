"""
File to create the Random Forest models.
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics
import cPickle
from Functions import feature_prune

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf("%s.h5" % datafile)
model_choice = raw_input("Regressor or Classifier (r\c\\nu): ")
save_deci = raw_input("Save model (y/n): ")
modelname = None
if save_deci == 'y':
    modelname = raw_input("filename: ")
dtree = None

if model_choice == 'r':
    from sklearn.ensemble import RandomForestRegressor
    y = X.pop('Distance 0')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    dtree = RandomForestRegressor(n_estimators=15)
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    print(metrics.mean_absolute_error(y_test, y_pred))
if model_choice == 'nu':
    from sklearn.ensemble import RandomForestRegressor
    y = X.pop('Nu')
    feature_prune(X, ['Distance 0'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    dtree = RandomForestRegressor(n_estimators=15)
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    print(metrics.mean_absolute_error(y_test, y_pred))
    a = y_test.tolist()
    b = y_pred.tolist()
    print type(y_test), type(y_pred)
# label = ['Target 0', 't1.0', 't1.5', 't2.0', 't2.5', 't3.0',
#        't3.5', 't4.0', 't4.5', 't5.0', 't5.5','t6.0', 't6.5', 't7.0',
#        't7.5', 't8.0', 't8.5', 't9.0', 't9.5']
label = ['Target 0']
Z = X.copy()
for i in label:
    X = Z.copy()
    if model_choice == 'c':
        from sklearn.ensemble import RandomForestClassifier
        y = X.pop(i)
        y = y.astype(int)
        temp = label[:]
        temp.remove(i)
        #print temp
        feature_prune(X, temp)
        # feature_prune(X, ['Distance 0'])
        #print X.keys()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        dtree = RandomForestClassifier(n_estimators=15)
        dtree.fit(X_train, y_train)
        y_pred = dtree.predict(X_test)
        final_frame = pd.DataFrame({'Test': y_test, 'Prediction': y_pred})
        print final_frame
        print(metrics.classification_report(y_test, y_pred))
        print(metrics.confusion_matrix(y_test, y_pred))

print '\n'
if save_deci == 'y':
    with open('%s.p' % modelname, 'wb') as f:
        cPickle.dump(dtree, f)
else:
    pass
