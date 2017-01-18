"""
Forms the FC plot for the confusion matrix. Feed in a Classifier dataframe and it's original dataframe.
"""

import pandas as pd
import numpy as np
from Functions import fcplot
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from Functions import binplot

datafile = raw_input("Classifier Pandas dataframe to open: ")
bin_datafile = raw_input("Original Pandas dataframe to open: ")
X = pd.read_hdf("%s.h5" % datafile)
B = pd.read_hdf("%s.h5" % bin_datafile)
y = X.pop('Target 0')
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.ensemble import RandomForestClassifier
dtree = RandomForestClassifier(n_estimators=15)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)


def confusion_conditions():
    values = []
    comparison_frame = pd.DataFrame({'Test': y_test, 'Prediction': y_pred})
    test_values = comparison_frame['Test'].values
    prediction_values = comparison_frame['Prediction'].values
    for i in range(len(comparison_frame.index.values)):

        test = test_values[i]
        prediction = prediction_values[i]

        # True Positive
        if test == 1 and prediction == 1:
            values.append(0)
        # False Negative
        if test == 1 and prediction == 0:
            values.append(3)
        # True Negative
        if test == 0 and prediction == 0:
            values.append(2)
        # False Positive
        if test == 0 and prediction == 1:
            values.append(1)

    confusion_series = pd.Series(np.array(values))
    print confusion_series.value_counts()
    print(metrics.confusion_matrix(y_test, y_pred))
    return confusion_series.values

confusion_series_values = confusion_conditions()
B['Confusion'] = pd.Series(np.array(confusion_series_values), index=y_test.index)
print B.columns
print B.head(5)

binplot(B, 'Confusion', condition= np.array(y_test.index.values)[np.array(B['Confusion'])[y_test.index.values] < 1.5 ])
