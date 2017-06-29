import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_palette('coolwarm')

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings('ignore')

# load train and test data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# remove the previously identified outlier
train = train.drop(883, axis=0)

(train.shape, test.shape)
# encode categorical data
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# get X and y for training
X_train = train.drop('y', axis=1)
y_train = train['y']

# get X for testing
X_test = test
X_train.shape, y_train.shape, X_test.shape

# PCA
n_comp = 2
pca = PCA(n_components=n_comp, random_state=420)
pca2d = pca.fit_transform(train.drop(['ID'], axis=1))

# plot PCA, notice an outlier
plt.scatter(pca2d[:, 0], pca2d[:, 1])
plt.show()

# identify outlier's index, value and ID
for i, j, k in zip(range(len(pca2d)), pca2d[:, 1], train['ID']):
    if j > 75:
        print(i, j, k)  # 883 154.822581442 1770

# KMeans clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit(train.drop(['ID'], axis=1))

# Plot PCA with clusters
colormap = np.array(sns.color_palette('Paired', 4).as_hex())
plt.scatter(pca2d[:, 0], pca2d[:, 1], c=colormap[clusters.labels_])
plt.show()

# Plot PCA, y, clusters
plt.scatter(train['y'], pca2d[:, 1], c=colormap[clusters.labels_])
plt.show()
