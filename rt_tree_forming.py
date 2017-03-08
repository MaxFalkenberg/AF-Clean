import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics
from sklearn.externals import joblib
from Functions import y_vector_classifier, x_vector_classifier

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf("%s.h5" % datafile)
model_choice = raw_input("Regressor or Classifier (r/c): ")
target = raw_input("y or x (y/x): ")

save_deci = raw_input("Save model (y/n): ")
modelname = None
if save_deci == 'y':
    modelname = raw_input("filename: ")
dtree = None

if model_choice == 'r':
    from sklearn.ensemble import RandomForestRegressor
    if target == 'y':
        # Prunes all the probe features not to do with Y vector
        probe_features = ['index', 'Target 0', 'Multi Target 0', 'Vector X 0', 'Theta 0', 'Distance 0', 'Nu', 'X Axis True', 'Y Axis True', 'Circuit True']
        all_features = X.columns
        for feature in probe_features:
            if feature in all_features:
                del X['%s' % feature]

        y = X.pop('Vector Y 0')

    if target == 'x':
        # Prunes all the probe features not to do with Y vector

        X = X[np.abs(X['Vector Y 0']) < 3]
        probe_features = ['index', 'Target 0', 'Multi Target 0', 'Vector Y 0', 'Theta 0', 'Distance 0', 'Nu','X Axis True', 'Y Axis True', 'Circuit True']
        all_features = X.columns
        for feature in probe_features:
            if feature in all_features:
                del X['%s' % feature]
        y = X.pop('Vector X 0')

    print "Training"
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    dtree = RandomForestRegressor(n_estimators=15)
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    print(metrics.mean_absolute_error(y_test, y_pred))

if model_choice == 'c':
    output_choice = raw_input("Predict or Probabilities (pre\pro): ")
    from sklearn.ensemble import RandomForestClassifier
    if target == 'y':
        X = X[np.array(X['X Axis True'])]
        probe_features = ['index', 'Target 0', 'Multi Target 0', 'Vector X 0','X Axis True','Vector Y 0', 'Theta 0', 'Distance 0', 'Nu', 'Y Axis True']
        all_features = X.columns
        for feature in probe_features:
            if feature in all_features:
                del X['%s' % feature]

        y = X.pop('Vector Y 0')
        # y = y.apply(y_vector_classifier)
        y = y.astype(int)

    if target == 'x':
        X = X[np.abs(X['Vector Y 0']) < 3]
        probe_features = ['index', 'Target 0', 'Multi Target 0', 'Vector Y 0', 'Theta 0', 'Distance 0', 'Nu']
        all_features = X.columns
        for feature in probe_features:
            if feature in all_features:
                del X['%s' % feature]

        y = X.pop('Vector X 0')
        # y = y.apply(x_vector_classifier)
        y = y.astype(int)

    print "Training"
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
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
    # with open('%s.p' % modelname, 'wb') as f:
    joblib.dump(dtree, "%s.pkl" % modelname)
else:
    pass
