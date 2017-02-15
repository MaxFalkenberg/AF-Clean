"""
Forms a fc plot for the class probabilties. Input is a Classifier dataframe and the original dataframe.
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from Functions import distance
from Functions import binplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

datafile = raw_input("Classifier Pandas dataframe to open: ")
bin_datafile = raw_input("Original Pandas dataframe to open: ")
thresholds = [5, 10, 15, 20]
# Output list:
# probm - probabilty map
# r_probm - probabilty map with diffrent thresholds
# probsl - probabilty slices
# hist - probabilty histogram

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
    output_figure = raw_input("Output figure (probm, e_probm, r_probm, probsl, e_probsl r_probsl, hist): ")
    if output_figure in ["probm", "e_probm", "r_probm", "probsl", "e_probsl", "r_probsl", "hist"]:
        break

X = pd.read_hdf("%s.h5" % datafile)
B = pd.read_hdf("%s.h5" % bin_datafile)

if output_figure in ["e_probm", "e_probsl"]:
    # Removes the target column as not needed
    if 'Target 0' in X.columns:
        del X['Target 0']
    if 'Multi Target 0' in X.columns:
        del X['Multi Target 0']
    y_scale_input = float(raw_input("y scale: "))
    x_scale_input = float(raw_input("x scale: "))
    y = pd.Series(distance(B['Vector X 0'], B['Vector Y 0'], y_scale=y_scale_input, x_scale=x_scale_input))
    y = y.apply(lambda x: 1 if x <= np.sqrt(200) else 0)
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    ML.fit(X_train, y_train)
    y_pred = ML.predict(X_test)
    y_prob = ML.predict_proba(X_test)
    on_cp_prob = [probs[1] for probs in y_prob]
    B['Positive Crit Prob'] = pd.Series(np.array(on_cp_prob), index=y_test.index.values)

if output_figure in ["probm", "probsl", "hist"]:
    if 'Target 0' in X.columns:
        y = X.pop('Target 0')
    if 'Multi Target 0' in X.columns:
        y = X.pop('Multi Target 0')
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    ML.fit(X_train, y_train)
    y_pred = ML.predict(X_test)
    y_prob = ML.predict_proba(X_test)
    on_cp_prob = [probs[1] for probs in y_prob]
    B['Positive Crit Prob'] = pd.Series(np.array(on_cp_prob), index=y_test.index.values)

if output_figure in ["r_probm", "r_probsl"]:
    cross_ref = X.pop('Distance 0')
    # threshold = float(raw_input('Threshold radius value: '))
    binned_dict = {}
    for t in thresholds:
        y = cross_ref.apply(lambda x: 1.0 if x <= t else 0.0)
        y = y.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        ML_ = ML
        ML_.fit(X_train, y_train)
        y_pred = ML_.predict(X_test)
        y_prob = ML_.predict_proba(X_test)
        on_cp_prob = [probs[1] for probs in y_prob]
        B['Positive Crit Prob'] = pd.Series(np.array(on_cp_prob), index=y_test.index.values)
        binned_grid, clim, feature = binplot(B, 'Positive Crit Prob', condition=y_test.index.values)
        binned_dict[t] = (binned_grid, clim, feature)

if output_figure in ["probm", "e_probm"]:
    # ret = False for probabilty map.
    binned_grid, clim, feature = binplot(B, 'Positive Crit Prob', condition=y_test.index.values)
    plt.figure(figsize=(10., 10.))
    cm = plt.cm.get_cmap('brg')
    plt.imshow(binned_grid, vmin=clim[0], vmax=clim[1], interpolation="nearest", origin="lower", cmap=cm)
    plt.colorbar(shrink=0.4, pad=0.07)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.title(feature, fontsize=18)
    plt.show()
    plt.close()

if output_figure == 'r_probm':
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for ax, t in zip(axes.flat, thresholds):
        binned_grid, clim, feature = binned_dict[t]
        im = ax.imshow(binned_grid, vmin=0, vmax=1, interpolation="nearest", origin="lower")
        ax.set_title(str(t), fontsize=18)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    plt.close()

if output_figure in ["probsl", "e_probsl"]:
    binned_grid, _, _ = binplot(B, 'Positive Crit Prob', condition=y_test.index.values)
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
    plt.title("Probabilty Slice through centre (constant x)")
    plt.xlabel("y")
    plt.ylabel("Probabilty for positive output (%s)" % model)
    plt.figure(2)
    plt.plot(range(len(y_slice)), y_slice)
    plt.title("Probabilty Slice through centre (constant y)")
    plt.xlabel("x")
    plt.ylabel("Probabilty for positive output (%s)" % model)
    plt.show()
    plt.close()

if output_figure == "r_probsl":
    print 'reached'
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.7])
    ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.7])
    x_slice_dict = {}
    y_slice_dict = {}
    for t in thresholds:
        binned_grid, clim, feature = binned_dict[t]
        processed_grid = np.nan_to_num(binned_grid)
        flat_processed_list = [prob for prob_list in processed_grid for prob in prob_list]
        # y_slice through grid (constant y)
        if processed_grid.shape[0] % 2 == 0:
            y_slice_index = processed_grid.shape[0]/2 - 1
            y_slice = processed_grid[y_slice_index]
            y_slice_dict[t] = y_slice
        elif processed_grid.shape[0] % 2 != 0:
            y_slice_index_1 = (processed_grid.shape[0] + 1)/2
            y_slice_index_2 = (processed_grid.shape[0] - 1)/2
            y_slice_1 = processed_grid[y_slice_index_1]
            y_slice_2 = processed_grid[y_slice_index_2]
            y_slice = [(x+y)/2. for x, y in zip(y_slice_1, y_slice_2)]
            y_slice_dict[t] = y_slice
        # x_slice throgh grid (constant x)
        if processed_grid.shape[1] % 2 == 0:
            x_slice_index = processed_grid.shape[1]/2 - 1
            x_slice = [sample[x_slice_index] for sample in processed_grid]
            x_slice_dict[t] = x_slice
        elif processed_grid.shape[1] % 2 != 0:
            x_slice_index_1 = (processed_grid.shape[1] + 1) / 2
            x_slice_index_2 = (processed_grid.shape[1] - 1) / 2
            x_slice_1 = [sample[x_slice_index_1] for sample in processed_grid]
            x_slice_2 = [sample[x_slice_index_2] for sample in processed_grid]
            x_slice = [(x+y)/2. for x,y in zip(x_slice_1,x_slice_2)]
            x_slice_dict[t] = x_slice

        ax1.plot(range(len(x_slice)), x_slice, label='Threshold: ' + str(t))
        ax1.set_title("Constant x")
        ax1.set_xlabel("y")
        ax1.set_ylabel("Probabilty for positive output (%s)" % model)
        ax2.plot(range(len(y_slice)), y_slice, label='Threshold: ' + str(t))
        ax2.set_title("Constant y")
        ax2.set_xlabel("x")
        ax2.set_ylabel("Probabilty for positive output (%s)" % model)

    plt.legend(loc='upper right')
    plt.show()
    plt.close()

if output_figure == "hist":
    binned_grid = binplot(B, 'Positive Crit Prob', condition=y_test.index.values)
    processed_grid = np.nan_to_num(binned_grid)
    flat_processed_list = [prob for prob_list in processed_grid for prob in prob_list]
    plt.hist(flat_processed_list, bins=50, range=[0.2, 1])
    plt.title("Histogram of the probabilty map.")
    plt.ylabel("Count")
    plt.xlabel("Bins")
    plt.show()
