"""
Forms a fc plot for the class probabilties. Input is a Classifier dataframe and the original dataframe.
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from Functions import binplot
from sklearn.ensemble import RandomForestClassifier

datafile = raw_input("Classifier Pandas dataframe to open: ")
bin_datafile = raw_input("Original Pandas dataframe to open: ")
# Output list:
# probm - probabilty map
# r_probm - probabilty map with diffrent thresholds
# probsl - probabilty slices
# hist - probabilty histogram

while True:
    output_figure = raw_input("Output figure (probm, r_probm, probsl, hist): ")
    if output_figure in ["probm", "r_probm", "probsl", "hist"]:
        break

X = pd.read_hdf("%s.h5" % datafile)
B = pd.read_hdf("%s.h5" % bin_datafile)

if output_figure in ["probm", "probsl", "hist"]:
    y = X.pop('Target 0')
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    dtree = RandomForestClassifier(n_estimators=15)
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    y_prob = dtree.predict_proba(X_test)
    on_cp_prob = [probs[1] for probs in y_prob]
    B['Positive Crit Prob'] = comparison_frame = pd.Series(np.array(on_cp_prob), index=y_test.index.values)

if output_figure == 'r_probm':
    cross_ref = X.pop('Distance 0')
    threshold = float(raw_input('Threshold radius value: '))
    # thresholds = [5, 6, 7, 8]
    # binned_dict = {}
    # for t in thresholds:
    y = cross_ref.apply(lambda x: 1.0 if x <= threshold else 0.0)
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.25)
    dtree = RandomForestClassifier(n_estimators=15)
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    y_prob = dtree.predict_proba(X_test)
    on_cp_prob = [probs[1] for probs in y_prob]
    B['Positive Crit Prob'] = comparison_frame = pd.Series(np.array(on_cp_prob), index=y_test.index.values)
    binned_grid, clim, feature = binplot(B, 'Positive Crit Prob', condition=y_test.index.values, ret=False)
    plt.figure(figsize=(10., 10.))
    cm = plt.cm.get_cmap('brg')
    plt.imshow(binned_grid, vmin=0, vmax=1, interpolation="nearest", origin="lower", cmap=cm)
    plt.colorbar(shrink=0.4, pad=0.07)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.title(feature, fontsize=18)
    plt.show()
    # binned_dict[t] = (binned_grid, clim, feature)

    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # for ax, t in axes.flat, thresholds:
    #     binned_grid, clim, feature = binned_dict[t]
    #     print binned_grid
    #     print clim
    #     print feature
    #     im = ax.imshow(binned_grid, vmin=0, vmax=1, interpolation="nearest", origin="lower")
    #     plt.title(feature, fontsize=18)
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    # plt.show()


if output_figure == "probm":
    # ret = False for probabilty map.
    binned_grid, clim, feature = binplot(B, 'Positive Crit Prob', condition=y_test.index.values, ret=False)
    plt.figure(figsize=(10., 10.))
    cm = plt.cm.get_cmap('brg')
    plt.imshow(binned_grid, vmin=clim[0], vmax=clim[1], interpolation="nearest", origin="lower", cmap=cm)
    plt.colorbar(shrink=0.4, pad=0.07)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.title(feature, fontsize=18)
    plt.show()

else:
    # ret = False for slice and hist.
    binned_grid = binplot(B, 'Positive Crit Prob', condition=y_test.index.values, ret=True)
    processed_grid = np.nan_to_num(binned_grid)
    flat_processed_list = [prob for prob_list in processed_grid for prob in prob_list]

if output_figure == "probsl":
    processed_grid = np.nan_to_num(binned_grid)
    flat_processed_list = [prob for prob_list in processed_grid for prob in prob_list]
    # y_slice through grid (constant y)
    if processed_grid.shape[0] % 2 == 0:
        y_slice_index = processed_grid.shape[0]/2 - 1
        y_slice = processed_grid[y_slice_index]
    elif processed_grid.shape[0] % 2 != 0:
        y_slice_index_1 = (processed_grid.shape[0] + 1)/2
        y_slice_index_2 = (processed_grid.shape[0] - 1)/2
        y_slice_1 = processed_grid[y_slice_index_1]
        y_slice_2 = processed_grid[y_slice_index_2]
        y_slice = [(x+y)/2. for x, y in zip(y_slice_1, y_slice_2)]

    # x_slice throgh grid (constant x)
    if processed_grid.shape[1] % 2 == 0:
        x_slice_index = processed_grid.shape[1]/2 - 1
        x_slice = [sample[x_slice_index] for sample in processed_grid]
    elif processed_grid.shape[1] % 2 != 0:
        x_slice_index_1 = (processed_grid.shape[1] + 1) / 2
        x_slice_index_2 = (processed_grid.shape[1] - 1) / 2
        x_slice_1 = [sample[x_slice_index_1] for sample in processed_grid]
        x_slice_2 = [sample[x_slice_index_2] for sample in processed_grid]
        x_slice = [(x+y)/2. for x,y in zip(x_slice_1,x_slice_2)]

    plt.figure(1)
    plt.plot(range(len(x_slice)), x_slice)
    plt.title("Probabilty Slice through centre (constant x) [SS_2000itt_n_nup2_Cup.h5]")
    plt.xlabel("y")
    plt.ylabel("Probabilty for positive output (RandomForest)")
    plt.figure(2)
    plt.plot(range(len(y_slice)), y_slice)
    plt.title("Probabilty Slice through centre (constant y) [SS_2000itt_n_nup2_Cup.h5]")
    plt.xlabel("x")
    plt.ylabel("Probabilty for positive output (RandomForest)")
    plt.show()

if output_figure == "hist":
    plt.hist(flat_processed_list, bins=50)
    plt.title("Histogram of the probabilty map.")
    plt.ylabel("Count")
    plt.xlabel("Bins")
    plt.show()