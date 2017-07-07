import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#% matplotlib inline
import seaborn as sns
sns.set_palette('Spectral')
import time
import os

# make np.seed fixed
seed=420
np.random.seed(seed)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
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

# import the packages
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping, ModelCheckpoint

# load train and test data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# Select only best features
features = ['X0',
                'X5',
                'X118',
                'X127',
                'X47',
                'X315',
                'X311',
                'X179',
                'X314',
                'X232',
                'X29',
                'X263',
                'X261']
train = train[features + ['ID', 'y']]
test = test[features + ['ID']]

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

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)

# number of samples
n_train, n_feats = X_train.shape
n_test, _ = X_test.shape

# define metrics function
from keras import backend as K


def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# reload model
model_path = 'output/model_keras3_resumed.h5'
model = load_model(model_path, custom_objects={'r2_keras': r2_keras})

# compile model
# model.compile(optimizer='rmsprop',
#               loss='MSE',
#               metrics=[r2_keras, 'accuracy'])
print(model.summary())

# train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2
)

callbacks = [
    EarlyStopping(
        monitor='val_r2_keras',
        patience=100,
        mode='max',
        verbose=1),
    ModelCheckpoint(
        model_path,
        monitor='val_r2_keras',
        save_best_only=True,
        mode='max',
        verbose=1)]

#%% fit the model with validation
hist = model.fit(X_tr,
                 y_tr,
                 epochs=2000,
                 #validation_split=0.2,
                 validation_data=(X_val, y_val),
                 batch_size=32,
                 verbose=2,
                 callbacks=callbacks
                 )
print(hist.history)

#save model to file
model.save('output/model_keras3_resumed.h5')

#%%
# Plot R^2
plt.figure()
plt.plot(hist.history['r2_keras'])
plt.plot(hist.history['val_r2_keras'])
plt.title('model accuracy')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('r2_keras_valdata_batch32_extrafeats_resume_savebest.png')

# predict
y_pred = model.predict(X_test, batch_size=1).ravel()

# create submission csv file
dirname = 'output'
count = len(os.listdir(os.path.join(os.getcwd(), dirname))) + 1
filename = 'sub' + str(count) + '_keras_extrafeats_savebest' + '.csv'
pd.concat([test.ID, pd.Series(y_pred)], axis=1).to_csv(dirname + '/' + filename,
                                                       header=['ID', 'y'], index=False)

os.system('say "finished running code"')