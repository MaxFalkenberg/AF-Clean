"""
File which performs hierachal clustering and plots dendrograms. Pandas classifier dataframes should be entered as input.
"""

import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from Functions import modeplot
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import random
import time

datafile = raw_input("Pandas Dataframe to load: ")
original_datafile = raw_input("Original Dataframe to load: ")
X = pd.read_hdf("%s.h5" % datafile)
M = pd.read_hdf("%s.h5" % original_datafile)

del X['Target 0']

rows = random.sample(X.index, 25000)
X = X.ix[rows]
data = scale(X.as_matrix().astype(float))

# Should read into what 'ward' does exactly.
# Default is euclidean.

start = time.time()
# Ward is the method for agglomerative clustering
Z = linkage(data, 'ward')
end = time.time()
print ('time passed: %s seconds' % (end - start))

plt.figure()
dendrogram(Z, no_labels=False, truncate_mode='lastp', p=16, show_contracted=False)
plt.show(block=False)


def clusterplot():
    cut_off = int(raw_input('Maximum d: '))
    results = fcluster(Z, cut_off, criterion='distance')
    M['Cluster Labels'] = pd.Series(results, index=rows)

#test case to check that cluster results are saved in dataframe correctly.
# for index, value in enumerate(rows):
#     if results[index] != M['Cluster Labels'][value]:
#         print "Test Failed"
#     else:
#         print "Passed"

# print results[0]
# print rows[0]
# print M['Cluster Labels'][rows[0]]

    grid, clim, feature = modeplot(M, 'Cluster Labels', condition=rows)
    plt.figure()
    cm = plt.cm.get_cmap('brg')
    plt.imshow(grid, vmin=clim[0], vmax=clim[1], interpolation="nearest", origin="lower", cmap=cm)
    # plt.colorbar(shrink=0.4, pad=0.07)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.title('%s - Maximum Distance: %s' %(feature, cut_off), fontsize=18)
    plt.show(block=False)

while True:
    clusterplot()
    replot = raw_input("Replot? (y/n): ")
    if replot == 'y':
        pass
    if replot != 'y':
        break

plt.close()