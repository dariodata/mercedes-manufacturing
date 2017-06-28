import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('muted')
import time
import os

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


class StackingEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)

        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prediction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed

# load train and test data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

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

n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(X_train)
tsvd_results_test = tsvd.transform(X_test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(X_train)
pca2_results_test = pca.transform(X_test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(X_train)
ica2_results_test = ica.transform(X_test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(X_train)
grp_results_test = grp.transform(X_test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(X_train)
srp_results_test = srp.transform(X_test)

#save columns list before adding the decomposition components

usable_columns = list(set(train.columns) - set(['y']))

# create empty dataframes to capture extra features
extra_features_train = pd.DataFrame()
extra_features_test = pd.DataFrame()

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    extra_features_train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    extra_features_test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    extra_features_train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    extra_features_test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    extra_features_train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    extra_features_test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    extra_features_train['grp_' + str(i)] = grp_results_train[:, i - 1]
    extra_features_test['grp_' + str(i)] = grp_results_test[:, i - 1]

    extra_features_train['srp_' + str(i)] = srp_results_train[:, i - 1]
    extra_features_test['srp_' + str(i)] = srp_results_test[:, i - 1]

extra_features_train.shape, extra_features_test.shape


X_train = np.hstack((X_train, extra_features_train))
X_test = np.hstack((X_test, extra_features_test))

# train XGB
print('Training')
t0 = time.time()
param = {'max_depth':2,
         'min_child_weight':4,
         'silent':1, 'objective':'reg:linear',
         'subsample':0.75
         }
dtrain = xgb.DMatrix(data=X_train, label=y_train)
#watchlist  = [(dtest,'test'), (dtrain,'train')]
xgb_model = xgb.train(param, dtrain)#, early_stopping_rounds=5)
print("Done: {:.1f} s".format(time.time() - t0))
os.system('say "finished training"')

def evalcoef(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

# train XGB
print('Cross validating')
t0 = time.time()
# param same as for training
# dtrain = xgb.DMatrix(data=X_train, label=y_train)
# watchlist  = [(dtest,'test'), (dtrain,'train')]
num_folds = 10
xgb_modelcv = xgb.cv(param, dtrain, num_folds, feval=evalcoef)#, early_stopping_rounds=5)
print("Done: {:.1f} s".format(time.time() - t0))
os.system('say "finished running code"')

print(xgb_modelcv[['test-r2-mean', 'train-r2-mean']])

# get predictions from model
dtest = xgb.DMatrix(data=X_test)
y_pred = xgb_model.predict(dtest)

# create submission csv file
dirname = 'output'
count = len(os.listdir(os.path.join(os.getcwd(), dirname))) + 1
filename = 'sub' + str(count) + '_xgb' + '_maxdepth' + str(param['max_depth'])  \
            + '.csv'
pd.concat([test.ID, pd.Series(y_pred)], axis=1).to_csv(dirname + '/' + filename,
                                                       header=['ID', 'y'], index=False)