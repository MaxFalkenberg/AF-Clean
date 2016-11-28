import os
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from Functions import print_progress
import cPickle

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf(os.path.join('Dataframes', "%s.h5" % datafile))
y = X.pop('Target')
y = y.astype(int)

rows = 2
robustness_datagrid = np.zeros((rows, len(X.columns)))

pp = 0
print_progress(pp, rows, prefix='Progress:', suffix='Complete', bar_length=50)
for row in range(rows):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    forest = RandomForestClassifier(n_estimators=15)
    forest.fit(X_train, y_train)
    robustness_datagrid[row] = forest.feature_importances_
    pp += 1
    print_progress(pp, rows, prefix='Progress:', suffix='Complete', bar_length=50)

mean_importance = np.mean(robustness_datagrid, axis=0)

sorted_mean_importance = np.sort(mean_importance)
sorted_std_importance = np.sort(np.std(robustness_datagrid, axis=0))
indicies = np.argsort(mean_importance)[::1]
feature_names_sorted = [X.columns[ind] for ind in indicies]

MY_dIR = os.path.realpath(os.path.dirname(__file__))
PICKLE_DIR = os.path.join(MY_dIR, 'Data')
fname = os.path.join(PICKLE_DIR, '%s_robust.p' % datafile)
with open(fname, 'wb') as f:
    cPickle.dump((sorted_mean_importance, sorted_std_importance, feature_names_sorted), f)

# fig = sns.plt.figure()
# sns.set(style="white")
# sns.barplot(sorted_mean_importance, feature_names_sorted, xerr=sorted_std_importance, color='r',
#             error_kw={'ecolor': 'k'})
# sns.plt.title("Feature Importance for %s" % datafile, fontsize=16)
# sns.plt.xlabel("Mean Feature Importance (Gini)", fontsize=14)
# sns.despine(left=True, offset=10)
# sns.plt.show()
# sns.plt.close()
