from keras.models import Model
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

def load_lstm_model(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape, name='time_input')
    const_input = Input(shape=const_input_shape, name='const_input')

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(10, return_sequences=False)(time_mask)

    # Concatenate the LSTM output with the constant input
    concat_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers
    dnn = layers.Dense(5, activation='relu')(concat_vector)
    life_pred = layers.Dense(1)(dnn)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

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

max_len = max(map(len, Xv))

Xv_train = pad_sequences(Xv_train, maxlen = max_len, padding='post', value = -999, dtype='float64')
Xv_test = pad_sequences(Xv_test, maxlen = max_len, padding='post', value = -999, dtype='float64')

model = load_lstm_model(Xv_train.shape[1:], Xc_train.shape[1:])

model.fit({"time_input": Xv_train, "const_input": Xc_train}, y_train, epochs=40, batch_size=5)
model.save('models/test_model2.h5')

scores = model.evaluate((Xv_test, Xc_test), y_test, verbose=0)

#%%
y_true = np.expm1(scaler_y.inverse_transform(y_test.reshape(-1, 1)))
y_pred = np.expm1(scaler_y.inverse_transform(model.predict((Xv_test, Xc_test)).reshape(-1, 1)))
rmse = mean_squared_error(y_true, y_pred)

rmse_scores.append(rmse)
print("{}: {:.2f}".format(model.metrics_names[1], rmse))

end = time.time()
print("Total time: {}".format(end - start))

print(abs(y_true-y_pred)/y_true*100)



# %%
