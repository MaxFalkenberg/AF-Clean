# SingleSource_ECGdata_Itt1000_P60_df
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sklearn.metrics as metrics
# from Functions import visualize_tree
from sklearn.externals import joblib

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf("%s.h5" % datafile)
del X['Target']

del X['Crit Position']
del X['Probe Position']
y = X.pop('Distance')
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

dtree = RandomForestRegressor()
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
# visualize_tree(dtree, feature_names=X_train.columns)
# print metrics.classification_report(y_test, y_pred)
# print metrics.confusion_matrix(y_test, y_pred)
print metrics.mean_absolute_error(y_test, y_pred)
joblib.dump(dtree, 'RFR1_Distance.pkl')
