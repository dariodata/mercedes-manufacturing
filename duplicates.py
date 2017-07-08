import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set_palette('Spectral')
# fix seed for reproducibility
seed = 420
np.random.seed(seed)

from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

from keras import backend as k
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import ELU

# load train and test data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# homogenize median of duplicates in train set
features = train.columns[2:]
cat_features = []
for c in train.columns:
    if train[c].dtype == 'object':
        cat_features.append(c)
duplicates = train[train.duplicated(subset=features, keep=False)].sort_values(by=cat_features)
medians = pd.read_csv('input/X0X118X127medians.csv')
medians = medians[medians.columns[:6]]
medians = medians.dropna(axis=0, subset=['y_median'])


def get_median(a, b, c):
    criterion1 = (medians['X0'] == a)
    criterion2 = (medians['X118'] == b)
    criterion3 = (medians['X127'] == c)
    return medians[criterion1 & criterion2 & criterion3].y_median.values[0]


def replace_median(df):
    df['y'] = get_median(df['X0'], df['X118'], df['X127'])
    return df


duplicates = duplicates.apply(lambda x: replace_median(x), axis=1)
train.loc[train.ID.isin(duplicates.ID), 'y'] = duplicates['y']

