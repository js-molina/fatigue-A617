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
from fatigue.neural.helper import load_known_lstm_model, preprocess_input
from fatigue.graph import chi_ratio
from fatigue.graph.models2 import graph_nn_prediction

start = time.time()
print("Starting timer...")

print('Loading Data...')
Xv, Xc, y = vectorise_data()

# Target Scaling

y = np.log1p(y)

Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test = train_test_split(Xv, Xc, y, random_state=1917)

Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
preprocess_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, max(map(len, Xv))) 

print('Creating Datasets...')

GPU = tf.config.list_logical_devices('GPU')

strategy = tf.distribute.MirroredStrategy(GPU)

# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

def tr_generator():
    for s1, s2, l in zip(Xv_train, Xc_train, y_train):
        yield {"time_input": s1, "const_input": s2}, l

def te_generator():
    for s1, s2, l in zip(Xv_test, Xc_test, y_test):
        yield {"time_input": s1, "const_input": s2}, l

train_data = tf.data.Dataset.from_generator(tr_generator,
                output_types=({"time_input": tf.float64, "const_input": tf.float64}, tf.float64))

train_data = train_data.shuffle(10000).batch(6)

test_data = tf.data.Dataset.from_generator(te_generator,
                output_types=({"time_input": tf.float64, "const_input": tf.float64}, tf.float64))

test_data = test_data.shuffle(10000).batch(6)

print('Initialising Model...')

with strategy.scope():
    model = load_known_lstm_model(Xv_train.shape[1:], Xc_train.shape[1:])

print('Fitting Model...')

model.fit(train_data, epochs=40)
    
model.save('models/' + 'test_model2.h5')

# # Inverse normalise target data

print('Evaluating Model...')

y_true = scaler_y.inverse_transform(y_test).reshape(-1)
y_pred = scaler_y.inverse_transform(model.predict(test_data)).reshape(-1)

# Inverse scale data
y_true, y_pred = map(np.expm1, [y_true, y_pred])

rmse = mean_squared_error(y_true, y_pred)

print("{}: {:.2f}".format(model.metrics_names[1], rmse))

err = abs(y_true-y_pred)/y_true*100

end = time.time()
print("Total time: {}".format((end - start)/60))

print(err.astype(np.float64))

np.savez('mdata/' + 'ydata_07_01_22' , y_obs=y_true, y_pred=y_pred)