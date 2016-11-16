"""
This will ask what HDF5 file you want loaded. The data will be stored in a nested dictionary.
The data is structured in groups, sub-groups and data-sets.

The group indicates the delta value
The sub-groups indicate the nu values
The data-sets contain the data (number of excited cells in grid for each time step).

groups are in the form: u'delta: 0.01' and the subgroups are in the form u'Nu: 0.13'
"""

import numpy as np
import h5py
from Functions import af_scan
from Functions import af_line_plot
from Functions import af_error_plot
from Functions import sampling_convert

print'\n'

print "Open Options: [Delta, Sampling, ML]"

choice = raw_input("Open type: ")

if choice == 'Delta':

    import matplotlib.pyplot as plt

    """Kishans Data sent by Kim (used for reference)"""
    kishans_nu = np.array([0.02, 0.04, 0.06, 0.08, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27,
                           0.29, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.1])

    kishans_values = np.array([0.99981, 0.99983, 0.9998, 0.99968, 0.99772, 0.96099, 0.60984, 0.16381, 0.017807,
                               0.020737, 4.922e-05, 0.0001084, 0, 0, 0.99152, 0.86184, 0.29714, 0.039206, 0.0056277,
                               4.834e-05, 0.00082172, 0, 0, 9.406e-05, 0.99919])

    file_name = raw_input("HDF5 to open: ")
    h5data = h5py.File('%s.h5' % file_name, 'r')
    h5par = h5py.File('%s_para.h5' % file_name, 'r')

    print('List of items in the base directory:', h5data.items())
    print('List of items in the base directory:', h5par.items())
    print'\n'

    # Displaying parameters on load.
    print "Parameters:"
    para = h5par.get('parameters')
    for i in para.iterkeys():
        print "%s: %s" % (i, np.array(para[i]))

    delta_range = np.array(para.get('Delta'))
    nu_range = np.array(para.get('Nu'))
    sim_size = np.array(para.get('Simulation Size'))
    iterations = np.array(para.get('Iterations'))
    pr = np.array(para.get('Pulse Rate'))

    print '\n'
    raw_data = {}  # This is where all the raw data is stored.

    for i in h5data.iterkeys():  # Structuring the data into total_data
        grp = h5data.get(i)
        key1 = '%s' % i
        raw_data[key1] = {}
        for j in grp.iterkeys():
            s_grp = grp.get(j)
            key2 = '%s' % j
            temp_data = []
            for k in s_grp.iterkeys():
                data = np.array(s_grp.get(k))
                temp_data.append(data)

            raw_data[key1][key2] = temp_data

    refined_data = {}  # Stores the data in tuples (mean, std)

    for delta in delta_range:
        refined_data['delta: %s' % delta] = {}
        for nu in nu_range:
            risk_data = np.zeros(iterations)
            for i in range(iterations):
                risk_data[i] = np.mean(np.array(af_scan(raw_data[u'delta: %s' % delta][u'Nu: %s' % nu][i], 200, pr)))
            grouped_risk_data = (np.mean(risk_data), np.std(risk_data))

            refined_data['delta: %s' % delta]['nu: %s' % nu] = grouped_risk_data

    """
    Plot for number of excited cells in each time step. If you want to plot these, need to change parameters in
    af_line_plot so that it uses the desired delta, nu, iteration. scanned finds where AF occurs.
    """
    plt.figure(1)
    af_line_plot(raw_data=raw_data, para=para, delta_=0.001, nu_=0.08, iteration=1, normalised=True)
    af_line_plot(0.001, 0.08, 1, normalised=False, scanned=True)
    plt.hlines((200 * 1.1)/float(
                max(raw_data[u'delta: %s' % 0.001][u'Nu: %s' % 0.08][1])), 0, sim_size, 'r', label='Threshold')
    plt.legend()
    plt.ylabel("Normalised number of excited cells")
    plt.xlabel("Time Step")

    """
    Plot showing the risk curve for different delta values. Kishans data is also plotted as reference.
    """
    plt.figure(2)
    for delta_values in delta_range:
        af_error_plot(delta_values, nu_range=nu_range, refined_data=refined_data, iterations=iterations)
    plt.plot(kishans_nu, kishans_values, 'r^', label="kishan")
    plt.grid(True)
    plt.ylabel("Risk of AF")
    plt.xlabel("Nu")
    plt.legend()
    plt.show()
    plt.close()

    h5data.close()
    h5par.close()

if choice == 'Sampling':

    import pyqtgraph as pg
    import pyqtgraph.ptime as ptime
    from pyqtgraph.Qt import QtCore, QtGui

    file_name = raw_input("HDF5 file to open: ")
    h5data = h5py.File("%s.h5" % file_name, 'r')
    h5par = h5py.File("%s_para.h5" % file_name, 'r')
    print '\n'

    for i in h5par.iterkeys():
        print "%s: %s" % (i, np.array(h5par['%s' % i]))

    print '\n'

    print range(np.array(h5par['Sample Interval']),
                np.array(h5par['Simulation Length']),
                np.array(h5par['Sample Interval']))

    print '\n'

    sample = int(raw_input("Please pick a sample to display: "))
    sample_range = range(np.array(h5par['Sample Range']) + np.array(h5par['Refractory Period']))
    sample_data = [np.array(h5data['Sample: %s' % sample]['dataset: %s' % i]) for i in sample_range]

    converted_data = list()
    grid = np.zeros((1000, 1000))
    sampling_convert(data=sample_data, output=converted_data, shape=grid.shape,
                     rp=np.array(h5par['Refractory Period']), animation_grid=grid)

    print len(converted_data)

    app = QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget()
    win.show()
    win.setWindowTitle('Sample: %s' % sample)
    view = win.addViewBox()
    view.setAspectLocked(True)

    view.setAspectLocked(True)
    img = pg.ImageItem(border='w')
    view.addItem(img)

    view.setRange(QtCore.QRectF(0, 0, 1000, 1000))
    animation_grid = np.zeros((1000, 1000))

    updateTime = ptime.time()
    fps = 0
    ptr = 49

    def update_data():
        global img, animation_grid, ptr, updateTime, fps
        ani_data = converted_data[ptr]
        # time.sleep(1/120.)  # gives larger more stable fps.
        img.setImage(ani_data.T, levels=(0, 50))
        ptr += 1

        if ptr == len(converted_data) - np.array(h5par['Refractory Period']) - 1:
            ptr = 49

        QtCore.QTimer.singleShot(1, update_data)
        now = ptime.time()
        fps2 = 1.0 / (now - updateTime)
        updateTime = now
        fps = fps * 0.9 + fps2 * 0.1

        # print "%0.1f fps" % fps

    update_data()

    if __name__ == '__main__':
        import sys

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    print "test"

if choice == 'ML':

    # SingleSource_ECGdata_Itt1000_P60_df
    import pandas as pd
    from sklearn.cross_validation import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from Functions import visualize_tree
    import seaborn as sns

    datafile = raw_input("Pandas dataframe to open: ")
    X = pd.read_hdf("%s.h5" % datafile)
    del X['Distance']
    del X['Crit Position']
    del X['Probe Position']
    y = X.pop('Target')
    y = y.astype(int)

    # Random state 1 (Need to change)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    print X_train.shape
    print X_test.shape
    print y_train.shape
    print y_test.shape

    dtree = DecisionTreeClassifier(random_state=0)
    dtree.fit(X_train, y_train)
    visualize_tree(dtree, feature_names=X_train.columns)
