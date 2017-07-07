import numpy as np
import pandas as pd
import os

y_pred3 = pd.read_csv('output/sub37_keras.csv')
y_pred2 = pd.read_csv('output/sub28_xgb_maxdepth4_eta0.0045_numround1250.csv')

y_pred = y_pred2['y']*.90 + y_pred3['y']*.10

# create submission csv file
dirname = 'output'
count = len(os.listdir(os.path.join(os.getcwd(), dirname))) + 1
filename = 'sub' + str(count) + '_ensemble_lasso_xgb_keras' + '.csv'
pd.concat([y_pred2.ID, pd.Series(y_pred)], axis=1).to_csv(dirname + '/' + filename,
                                                       header=['ID', 'y'], index=False)