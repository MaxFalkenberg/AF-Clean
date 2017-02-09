"""
File to create k-means clustering for ECG data. Input a classification dataframe.
Should input a regression database for FC plot.
"""

import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import Counter

# Question for PCA
while True:
    PCA_yn = raw_input("PCA (y/n): ")
    if PCA_yn in ['y', 'n']:
        break

# Question for Scaling
while True:
    if PCA_yn == 'y':
        scaled = 'y'
        break
    scaled = raw_input("Scale dataframes (y/n): ")
    if scaled in ['y', 'n']:
        break

# Number of clusters that the k-mean model will try to fit to.
n_clusters = 2

datafile = raw_input("Pandas dataframe to open: ")
dataframe = pd.read_hdf("%s.h5" % datafile)

# removes the classification features.
y = dataframe.pop('Target 0')
# Samples a random selection from the dataframe for plotting ease.
rows = random.sample(dataframe.index, 50000)
X = dataframe.ix[rows]
data = X.as_matrix().astype(float)

if scaled == 'y':
    # scales the data such that the std=1 for all features
    # (this creates the assumption that all features are important.)
    data = scale(X.as_matrix().astype(float))

# k-means modelgeneration
k_model = KMeans(n_clusters=n_clusters, init='k-means++')

# Simply gives back the silhouette score
if PCA_yn == 'n':
    cluster_centres = k_model.fit_predict(data)
    print silhouette_score(data, k_model.labels_, metric='euclidean', sample_size=300)

# Data should be scaled before PCA is performed (according to ISLR book)
# Plots the clusters and what the the samples are classified
if PCA_yn == 'y':
    # Decomposis the feature space to 2 priciple compenents (allows the clusters to be plotted).
    reduced_data = PCA(n_components=2).fit_transform(data)
    cluster_labels = k_model.fit_predict(reduced_data)
    cluster_centres = k_model.cluster_centers_
    # silhouette_score: -1 bad, 1 good
    print "Silhouette Score: %s" % silhouette_score(reduced_data, k_model.labels_, metric='euclidean', sample_size=300)

    targets = y.ix[rows]
    target_data = (targets.as_matrix().astype(float) / n_clusters)
    print Counter(target_data)

    colors1 = cm.hsv(cluster_labels.astype(float) / n_clusters)
    colors2 = cm.hsv(target_data)
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.7])
    ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.7])
    ax1.scatter(reduced_data[:, 0], reduced_data[:, 1], marker = '.', c=colors1, alpha=0.7)
    ax1.scatter(cluster_centres[:, 0], cluster_centres[:, 1], marker = 'o', c='white', alpha=1.0)
    ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], marker = '.', c=colors2, alpha=1)
    ax1.set_xlabel("First Principle Component")
    ax1.set_ylabel("Second Principle Component")
    ax2.set_xlabel("First Principle Component")
    ax2.set_ylabel("Second Principle Component")
    plt.show()
    plt.close()
