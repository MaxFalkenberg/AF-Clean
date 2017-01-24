"""
Forms a fc plot for the class probabilties. Input is a Classifier dataframe and the original dataframe.
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from Functions import binplot, fcplot

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
y_prob = dtree.predict_proba(X_test)

print y_prob
on_cp_prob = [probs[1] for probs in y_prob]
B['Positive Crit Prob'] = comparison_frame = pd.Series(np.array(on_cp_prob), index=y_test.index.values)

#Change ret to False for prob map.
binned_grid = binplot(B, 'Positive Crit Prob', condition=y_test.index.values, ret=True)
processed_grid = np.nan_to_num(binned_grid)
flat_processed_list = [prob for prob_list in processed_grid for prob in prob_list]

#Prob Slice
# # y_slice through grid (constant y)
# if processed_grid.shape[0] % 2 == 0:
#     y_slice_index = processed_grid.shape[0]/2 - 1
#     y_slice = processed_grid[y_slice_index]
# elif processed_grid.shape[0] % 2 != 0:
#     y_slice_index_1 = (processed_grid.shape[0] + 1)/2
#     y_slice_index_2 = (processed_grid.shape[0] - 1)/2
#     y_slice_1 = processed_grid[y_slice_index_1]
#     y_slice_2 = processed_grid[y_slice_index_2]
#     y_slice = [(x+y)/2. for x, y in zip(y_slice_1, y_slice_2)]
#
# # x_slice throgh grid (constant x)
# if processed_grid.shape[1] % 2 == 0:
#     x_slice_index = processed_grid.shape[1]/2 - 1
#     x_slice = [sample[x_slice_index] for sample in processed_grid]
# elif processed_grid.shape[1] % 2 != 0:
#     x_slice_index_1 = (processed_grid.shape[1] + 1) / 2
#     x_slice_index_2 = (processed_grid.shape[1] - 1) / 2
#     x_slice_1 = [sample[x_slice_index_1] for sample in processed_grid]
#     x_slice_2 = [sample[x_slice_index_2] for sample in processed_grid]
#     x_slice = [(x+y)/2. for x,y in zip(x_slice_1,x_slice_2)]
#
# plt.figure(1)
# plt.plot(range(len(x_slice)), x_slice)
# plt.title("Probabilty Slice through centre (constant x) [SS_2000itt_n_nup2_Cup.h5]")
# plt.xlabel("y")
# plt.ylabel("Probabilty for positive output (RandomForest)")
# plt.figure(2)
# plt.plot(range(len(y_slice)), y_slice)
# plt.title("Probabilty Slice through centre (constant y) [SS_2000itt_n_nup2_Cup.h5]")
# plt.xlabel("x")
# plt.ylabel("Probabilty for positive output (RandomForest)")
# plt.show()

# hist_data, bin_data = np.histogram(flat_processed_list, bins=10, range=(0,1))
# print hist_data
plt.hist(flat_processed_list, bins=50)
plt.title("Histogram of the probabilty map.")
plt.ylabel("Count")
plt.xlabel("Bins")
plt.show()