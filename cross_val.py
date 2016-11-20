# SingleSource_ECGdata_Itt1000_P60_df
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from Functions import print_progress
import matplotlib.pyplot as plt

datafile = raw_input("Pandas dataframe to open: ")
X = pd.read_hdf("%s.h5" % datafile)
del X['Distance']
del X['Crit Position']
del X['Probe Position']
y = X.pop('Target')
y = y.astype(int)

mean_scores_l = list()
# errors = list()
progress = 0
n_tree_range = range(1, 15)
print_progress(progress, len(n_tree_range), prefix='Progress', suffix='Complete', bar_length=50)
for i in n_tree_range:
    dtree = RandomForestClassifier(n_estimators=i)
    scores = cross_val_score(dtree, X, y, cv=10, scoring='accuracy')
    mean_scores_l.append(np.mean(scores))
    # errors.append(np.std(scores))
    progress += 1
    print_progress(progress, len(n_tree_range), prefix='Progress', suffix='Complete', bar_length=50)

plt.figure()
plt.title("K-Cross Validation for Random Forest Classifier (SingleSource_ECGdata_Itt1000_P60_df)")
plt.xlabel("Number of Tree Estimators")
plt.ylabel("Classification Accuracy")
plt.plot(n_tree_range, mean_scores_l)
plt.show()
