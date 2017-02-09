"""
File to create k-means clustering for ECG data. Input a classification dataframe.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

while True:
    PCA_yn = raw_input("PCA (y/n): ")
    if PCA_yn in ['y', 'n']:
        break

while True:
    scaled = raw_input("Scale dataframes (y/n): ")
    if scaled in ['y', 'n']:
        break

# Number of clusters that the k-mean model will try to fit to.
n_clusters = 2

datafile = raw_input("Pandas dataframe to open: ")
dataframe = pd.read_hdf("%s.h5" % datafile)

# removes the classification features.
y = dataframe.pop('Target 0')
X_train, X_test, y_train, y_test = train_test_split(dataframe, y, train_size=0.05)

data = X_train.as_matrix().astype(float)
if scaled == 'y':
    # scales the data such that the std=1 for all features (this creates the assumption that all features are important.)
    data = scale(X_train.as_matrix().astype(float))

k_model = KMeans(n_clusters=n_clusters, init='k-means++')

if PCA_yn == 'n':
    cluster_centres = k_model.fit_predict(data)
    print silhouette_score(data, k_model.labels_, metric='euclidean', sample_size=300)

if PCA_yn == 'y':
    # Decomposis the feature space to 2 priciple compenents (allows the clusters to be plotted).
    reduced_data = PCA(n_components=2).fit_transform(data)

    cluster_labels = k_model.fit_predict(reduced_data)
    cluster_centres = k_model.cluster_centers_
    # silhouette: -1 bad, 1 good
    print silhouette_score(reduced_data, k_model.labels_, metric='euclidean', sample_size=300)

    colors = cm.hsv(cluster_labels.astype(float) / n_clusters)
    fig, ax1 = plt.subplots()
    ax1.scatter(reduced_data[:, 0], reduced_data[:, 1], marker = '.', c=colors, alpha=0.7)
    ax1.scatter(cluster_centres[:, 0], cluster_centres[:, 1], marker = 'o', c='white', alpha=1.0)
    plt.xlabel("First Principle Component")
    plt.ylabel("Second Principle Component")
    plt.show()
