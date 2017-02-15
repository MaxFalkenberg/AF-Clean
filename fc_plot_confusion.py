"""
Forms the FC plot for the confusion matrix. Feed in a Classifier dataframe and it's original dataframe.
Also can save a range of scaling targets.
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from Functions import distance, modeplot, print_progress
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import pickle


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
    print(metrics.classification_report(y_test, y_pred))
    return confusion_series.values

datafile = raw_input("Pandas dataframe to open: ")
bin_datafile = raw_input("Original Pandas dataframe to open: ")

while True:
    model = raw_input("ML model (RF, GNB, LR): ")
    if model in ['RF', 'GNB', 'LR']:
        break

if model == 'RF':
    ML = RandomForestClassifier(n_estimators=15)

if model == 'GNB':
    ML = GaussianNB()

if model == 'LR':
    ML = LogisticRegression()

while True:
    threshold_type = raw_input("Threshold type (c,e,re): ")
    if threshold_type in ['c', 'e', 're']:
        break

X = pd.read_hdf("%s.h5" % datafile)
B = pd.read_hdf("%s.h5" % bin_datafile)

if threshold_type == 'c':
    y = X.pop('Target 0')
    y = y.astype(int)

if threshold_type == 'e':
    # Removes the target column as not needed
    remove_pop = X.pop('Target 0')
    y_scale_input = float(raw_input("y scale: "))
    x_scale_input = float(raw_input("x scale: "))
    y = pd.Series(distance(B['Vector X 0'], B['Vector Y 0'], y_scale=y_scale_input, x_scale=x_scale_input))
    y = y.apply(lambda x: 1 if x <= np.sqrt(200) else 0)
    y = y.astype(int)

if threshold_type == 're':
    y_scale_range = np.arange(1.0, 8.5, 0.5)
    x_scale_range = np.arange(0.5, 2.0, 0.5)
    results_dict = {}
    remove_pop = X.pop('Target 0')

    pp = 0
    print_progress(pp, len(y_scale_range), prefix='Progress:', suffix='Complete', bar_length=50)
    for y_scale in y_scale_range:
        for x_scale in x_scale_range:
            y = pd.Series(
                distance(B['Vector X 0'], B['Vector Y 0'], y_scale=y_scale, x_scale=x_scale))
            y = y.apply(lambda x: 1 if x <= np.sqrt(200) else 0)
            y = y.astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            ML_ = ML
            ML_.fit(X_train, y_train)
            y_pred = ML_.predict(X_test)
            confusion_mat = metrics.confusion_matrix(y_test, y_pred)
            results_dict[(y_scale, x_scale)] = confusion_mat

        pp += 1
        print_progress(pp, len(y_scale_range), prefix='Progress:', suffix='Complete', bar_length=50)

    with open('%s_conf_s.p' % datafile, 'wb') as f:
        pickle.dump(results_dict, f)

if threshold_type != 're':
    """
    Does the plotting if the threshold type is not a range.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    ML.fit(X_train, y_train)
    y_pred = ML.predict(X_test)

    confusion_series_values = confusion_conditions()
    B['Confusion'] = pd.Series(np.array(confusion_series_values), index=y_test.index)
    z, clim, feature = modeplot(B, 'Confusion', condition= np.array(y_test.index.values))
    # [np.array(B['Confusion'])[y_test.index.values] < 1.5 ])

    # Need to alter still
    # z_shape = z.shape
    # z = [l[i] for i in range(z.shape[1]) for l in z]
    # print z
    # z = [x if x >= 2.5 else float('NaN') for x in z]
    # z = np.array(z).reshape(z_shape)

    cm = plt.cm.get_cmap('brg')
    plt.figure(figsize =(10.,10.))
    plt.imshow(z,vmin = clim[0],vmax = clim[1], interpolation="nearest", origin="lower", cmap = cm)
    plt.colorbar(shrink=0.4, pad = 0.07)
    plt.xlabel('x', fontsize = 18)
    plt.ylabel('y', fontsize = 18)
    plt.title(feature, fontsize = 18)
    plt.show()
    plt.close()