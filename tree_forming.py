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
    # X = X[np.absolute(np.array(X['Vector Y 0'])) < 3.]
    y = X.pop('Vector Y 0')
    try:
        feature_prune(X, [ 'Target 0', 'Vector X 0', 'Multi Target 0', 'Nu', 'Theta 0', 'Distance 0', 'index'])
    except:
        print 'prune failed'
        pass
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
label = ['Vector Y 0']
Z = X.copy()
for i in label:
    X = Z.copy()
    # probe_features = ['Y Axis True 3','Y Axis True 4','Y Axis True 5', 'Y Axis True 6','Y Axis True 7','Y Axis True 8','Y Axis True 9','Y Axis True 10','Y Axis True 11','Y Axis True 12','Crit Position', 'Crit Position 0', 'Crit Position 1', 'Probe Position','Target 0',
    #                   'Unit Vector X', 'Unit Vector X 0', 'Unit Vector X 1', 'Unit Vector Y', 'Unit Vector Y 0',
    #                   'Unit Vector Y 1', 'Theta', 'Theta 0', 'Theta 1', 'Probe Number', 'Multi Target 0',
    #                   'Nearest Crit Position','Vector X 0','Vector Y 0','Sample Length','Nu', 'Distance 0']

    probe_features = ['Crit Position', 'Crit Position 0', 'Crit Position 1', 'Probe Position','Target 0',
                      'Unit Vector X', 'Unit Vector X 0', 'Unit Vector X 1', 'Unit Vector Y', 'Unit Vector Y 0',
                      'Unit Vector Y 1', 'Theta', 'Theta 0', 'Theta 1', 'Probe Number', 'Multi Target 0',
                      'Nearest Crit Position','Vector X 0','Sample Length','Nu', 'Distance 0']

    # Deletes features from the dataframe that are in probe_features
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
        X_test_meta = X_test[['Target 0', 'Vector X 0', 'Multi Target 0', 'Nu', 'Theta 0', 'Distance 0', 'index']]

        all_features = list(X_train.columns)
        for feature in probe_features:
            if feature in all_features:
                del X_train['%s' % feature]

        all_features = list(X_test.columns)
        for feature in probe_features:
            if feature in all_features:
                del X_test['%s' % feature]

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
