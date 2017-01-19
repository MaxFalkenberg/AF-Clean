"""
File to create the Random Forest models.
"""
import pandas as pd
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics
import cPickle

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf("%s.h5" % datafile)
model_choice = raw_input("Regressor or Classifier (r\c): ")
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

if model_choice == 'c':
    output_choice = raw_input("Predict or Probabilities (pre\pro): ")
    from sklearn.ensemble import RandomForestClassifier
    y = X.pop('Target 0')
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    dtree = RandomForestClassifier(n_estimators=15)
    dtree.fit(X_train, y_train)
    if output_choice == 'pre':
        y_pred = dtree.predict(X_test)
        final_frame = pd.DataFrame({'Test': y_test, 'Prediction': y_pred})
        print final_frame
        print(metrics.classification_report(y_test, y_pred))
        print(metrics.confusion_matrix(y_test, y_pred))
    if output_choice == 'pro':
        y_prob = dtree.predict_proba(X_test)
        y_pred = dtree.predict(X_test)
        print y_prob
        print y_pred

print '\n'
if save_deci == 'y':
    with open('%s.p' % modelname, 'wb') as f:
        cPickle.dump(dtree, f)
else:
    pass
