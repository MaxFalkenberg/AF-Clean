"""
Creates a logistic regression model machine learning model.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf("%s.h5" % datafile)
y = X.pop('Target 0')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
