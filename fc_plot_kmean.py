"""
File which forms an fc plot for samples which are clustered using the k-means method.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from collections import Counter
from Functions import binplot

# number of k-means clusters to form:
# Off the critical circuit
# On the critical circuit
# At a boundry point
n_clusters = 3

filename = raw_input("Pandas dataframe to open: ")
bin_datafile = raw_input("Original Pandas dataframe to open: ")
dataframe = pd.read_hdf("%s.h5" % filename)
B = pd.read_hdf("%s.h5" % bin_datafile)
y = dataframe.pop('Target 0')
# Scaling data for clustering (std = 1 for all features)
X = scale(dataframe.as_matrix().astype(float))
k_model = KMeans(n_clusters=n_clusters, init='k-means++')

cluster_labels = k_model.fit_predict(X)
B['Cluster Label'] = pd.Series(cluster_labels)

binned_grid, clim, feature = binplot(B, 'Cluster Label')
plt.figure()
cm = plt.cm.get_cmap('brg')
plt.imshow(binned_grid, vmin=clim[0], vmax=clim[1], interpolation="nearest", origin="lower", cmap=cm)
plt.colorbar(shrink=0.4, pad=0.07)
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.title(feature, fontsize=18)
plt.show()
