#%%
from keras.models import Model, load_model
import pandas as pd
import numpy as np
import seaborn as sb
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras import layers, Input, optimizers, losses, metrics, Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, PowerTransformer, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
import time
import datetime
from keras.wrappers.scikit_learn import KerasRegressor

from fatigue.networks import vectorise_data, ragged_numpy_arr

start = time.time()
print("Starting timer...")

Xv, Xc, y = vectorise_data()

target_scaling = True

if target_scaling:
    y = np.log1p(y)

rmse_scores = []
all_y_true = []
all_y_pred = []

Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test = train_test_split(Xv, Xc, y, random_state=69)

tempX = pd.concat(Xv_train).reset_index(drop=True)

scaler_var = PowerTransformer()
scaler_var.fit(tempX)

# Scaling and normalising data.

Xv_train = np.array([scaler_var.transform(x) for x in Xv_train], dtype='object')
Xv_test = np.array([scaler_var.transform(x) for x in Xv_test], dtype= 'object')

scaler_con = StandardScaler()
scaler_con.fit(Xc_train)

Xc_train = scaler_con.transform(Xc_train)
Xc_test = scaler_con.transform(Xc_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Normalising output data.
scaler_y = StandardScaler()
scaler_y.fit(y_train)

y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)

# Padding input sequences

max_len = max(map(len, Xv))

Xv_train = pad_sequences(Xv_train, maxlen = max_len, padding='post', value = -999, dtype='float64')
Xv_test = pad_sequences(Xv_test, maxlen = max_len, padding='post', value = -999, dtype='float64')

model = load_model('models/test_model2.h5')

y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
y_pred = scaler_y.inverse_transform(model.predict((Xv_test, Xc_test)).reshape(-1, 1)).reshape(-1)

y_true, y_pred = map(np.expm1, [y_true, y_pred])

rmse = mean_squared_error(y_true, y_pred)

print("{}: {:.2f}".format(model.metrics_names[1], rmse))

end = time.time()
print("Total time: {}".format(end - start))


print(abs(y_true-y_pred)/y_true*100)

# %%
