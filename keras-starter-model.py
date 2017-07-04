import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#% matplotlib inline
import seaborn as sns
sns.set_palette('Spectral')
import time
import os

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

# import the packages
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping

# load train and test data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
# encode categorical data
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))
# remove the previously identified outlier
train = train.drop(883, axis=0)
X_train = np.array(train.drop('y', axis=1))
y_train = np.array(train['y'])
X_test = np.array(test)

# number of samples
n_train, n_feats = X_train.shape
n_test, _ = X_test.shape

# define metrics function
from keras import backend as K
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#%% Set up the neural network

# build the neural network
model = Sequential()

model.add(Dense(n_feats, input_dim=n_feats, kernel_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(n_feats, kernel_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(n_feats//2, kernel_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(n_feats//2, kernel_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(n_feats//2, kernel_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(1))

#%% compile model
model.compile(optimizer='rmsprop',
              loss='MSE',
              metrics=[r2_keras, 'accuracy'])
print(model.summary())

#%%
#from keras.utils.visualize_util import plot
#plot(model, to_file='digit-recognizer/model.png')

callbacks = [
    EarlyStopping(
        monitor='val_r2_keras',
        patience=40,
        mode='max',
        verbose=1)]

#%% fit the model with validation
hist = model.fit(X_train, y_train, epochs=200, validation_split=0.2, batch_size=1, verbose=2, callbacks=callbacks)
print(hist.history)

#save model to file
model.save('output/model_keras1.h5')

#%%
# Plot R^2
plt.figure()
plt.plot(hist.history['r2_keras'])
plt.plot(hist.history['val_r2_keras'])
plt.title('model accuracy')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('r2_keras.png')


# # Plot accuracy
# plt.figure()
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('r2_keras.png')
#
# #%%
# # Plot loss
# plt.figure()
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('r2_keras.png')


#%% fit model without validation
#model.fit(X_train, y_train, nb_epoch=10, batch_size=32)

#%% evaluate on test data
#loss, accuracy = model.evaluate(X_test, y_test)
#print('loss:', loss)
#print('accuracy:', accuracy)

#%% predict
#model = load_model('digit-recognizer/model_kaggle.h5')
y_pred = model.predict(X_test, batch_size=1).ravel()

# create submission csv file
dirname = 'output'
count = len(os.listdir(os.path.join(os.getcwd(), dirname))) + 1
filename = 'sub' + str(count) + 'keras' + '.csv'
pd.concat([test.ID, pd.Series(y_pred)], axis=1).to_csv(dirname + '/' + filename,
                                                       header=['ID', 'y'], index=False)