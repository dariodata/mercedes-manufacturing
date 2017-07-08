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

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils import check_array
from sklearn.linear_model import LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder


import warnings
warnings.filterwarnings('ignore')

# Quickstart
# load train and test data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

##############################################
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
##############################################################

# encode categorical data
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))
# remove the previously identified outlier
#train = train.drop(883, axis=0)
X_train = train.drop('y', axis=1)
y_train = train['y']
X_test = test


class StackingEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)

        # add class probabilities as a synthetic additional feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prediction as a synthetic additional feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed

seed=420
stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3,
                                                          max_features=0.55, min_samples_leaf=18, min_samples_split=14,
                                                          subsample=0.7, random_state=seed)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    LassoLarsCV()
)

# stacked model is trained without the extra features
t0 = time.time()
stacked_pipeline.fit(X_train, y_train)
y_pred = stacked_pipeline.predict(X_test)
print("Done: {:.1f} s".format(time.time() - t0))


# create submission csv file
dirname = 'output'
count = len(os.listdir(os.path.join(os.getcwd(), dirname))) + 1
filename = 'sub' + str(count) + '_lasso_stacked' + '.csv'
pd.concat([test.ID, pd.Series(y_pred)], axis=1).to_csv(dirname + '/' + filename,
                                                       header=['ID', 'y'], index=False)