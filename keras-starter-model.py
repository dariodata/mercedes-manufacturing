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

'''
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
'''

# encode categorical data
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# remove the previously identified outlier
# train = train.drop(883, axis=0)

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

X_train = np.hstack((X_train, extra_features_train))
X_test = np.hstack((X_test, extra_features_test))

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)

# number of samples
n_train, n_feats = X_train.shape
n_test, _ = X_test.shape

print('Finished data pre-processing.')
print('Train set shape: ', X_train.shape)
print('Test set shape: ', X_test.shape)

print('Creating neural network...')

# build the neural network
model = Sequential()

model.add(Dense(n_feats, input_dim=n_feats))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(0.5))

model.add(Dense(n_feats))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(0.3))

model.add(Dense(n_feats))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(0.3))

model.add(Dense(n_feats // 2))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(0.3))

model.add(Dense(n_feats // 4))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(0.3))

model.add(Dense(1, activation='linear'))


def r2_keras(y_true, y_pred):
    """
    Custom metric function r^2 for accuracy
    :param y_true:
    :param y_pred:
    :return: r^2
    """
    SS_res = k.sum(k.square(y_true - y_pred))
    SS_tot = k.sum(k.square(y_true - k.mean(y_true)))
    return 1 - SS_res / (SS_tot + k.epsilon())


# compile model
model.compile(optimizer='adam',
              loss='MSE',
              metrics=[r2_keras, 'accuracy'])

print('Finished compiling model:')
print(model.summary())

# from keras.utils.visualize_util import plot
# plot(model, to_file='digit-recognizer/model.png')

# model path for saving model
previous_model_path = 'output/model_ELU_2dup_val_resumed.h5'
model_path = 'output/model_ELU_2dup_val_resumed2.h5'


# train/validation split
train_orig = pd.read_csv('input/train.csv')
X_tr, X_val, y_tr, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
_, _, _, y_val = train_test_split(X_train, train_orig['y'], test_size=0.2, random_state=0)

# define callbacks before fitting
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
        verbose=1)
]

print('Start model fitting...')
t0 = time.time()

if os.path.isfile(previous_model_path):
    # reload previously fit model to continue fitting
    model = load_model(previous_model_path, custom_objects={'r2_keras': r2_keras})

# fit the model with validation data
hist = model.fit(X_tr,
                 y_tr,
                 epochs=2000,
                 # validation_split=0.2,
                 validation_data=(X_val, y_val),
                 batch_size=32,
                 verbose=2,
                 callbacks=callbacks
                 )

print('Finished fitting model. Done: {:.1f}s'.format(time.time() - t0))
print(hist.history)
print('Maximum r2_keras value: {0}'.format(np.max(hist.history['val_r2_keras'])))

# save model to file
# model.save(model_path) # not necessary because of ModelCheckPoint callback

# Plot R^2
plt.figure()
plt.plot(hist.history['r2_keras'])
plt.plot(hist.history['val_r2_keras'])
plt.title('Model accuracy')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('output/r2_keras_ELU_2dup_val_resumed2.png')

# # Plot accuracy
# plt.figure()
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('r2_keras.png')

# # Plot loss
# plt.figure()
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('r2_keras.png')

# %% fit model without validation
# model.fit(X_train, y_train, nb_epoch=10, batch_size=32)

# evaluate on test data
# loss, accuracy = model.evaluate(X_test, y_test)
# print('loss:', loss)
# print('accuracy:', accuracy)

print('Loading saved best model...')

# if best iteration's model was saved then load and use it
if os.path.isfile(model_path):
    model = load_model(model_path, custom_objects={'r2_keras': r2_keras})

print('Making predictions...')

# predictions
y_pred = model.predict(X_test).ravel()

# create submission csv file
dirname = 'output'
count = len(os.listdir(os.path.join(os.getcwd(), dirname))) + 1
filename = 'sub' + str(count) + '_keras_ELU_dup_val_resumed2' + '.csv'
pd.concat([test.ID, pd.Series(y_pred)], axis=1).to_csv(dirname + '/' + filename,
                                                       header=['ID', 'y'], index=False)

print('Finished. Predictions saved as file: ', filename)
#os.system('say "finished running code"')
