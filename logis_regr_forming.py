"""
File which creates a Logistic Regression Classifier model.
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf("%s.h5" % datafile)
# save_deci = raw_input("Save model (y/n): ")
# modelname = None
# if save_deci == 'y':
#     modelname = raw_input("filename: ")

y = X.pop('Target 0')
X_train, X_test, y_train, y_test = train_test_split(X, y)

NB = GaussianNB()
NB.fit(X_train, y_train)
y_pred = NB.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
