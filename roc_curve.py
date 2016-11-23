# SingleSource_ECGdata_Itt1000_P60_df
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics
import seaborn as sns

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf(os.path.join('Data', "%s.h5" % datafile))
del X['Distance']
del X['Crit Position']
del X['Probe Position']
y = X.pop('Target')
y = y.astype(int)

# should put this somewhere where it opens a ML file
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
dtree = RandomForestClassifier(n_estimators=10)
dtree.fit(X_train, y_train)
prob_pred = dtree.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, prob_pred)
roc_auc = metrics.auc(fpr, tpr)
print roc_auc
sns.plt.plot(fpr, tpr)
sns.plt.show()