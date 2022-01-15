import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)

from keras.models import Model, load_model
import pandas as pd
import numpy as np
import seaborn as sb
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from keras import layers, Input, optimizers, losses, metrics, Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, PowerTransformer, Normalizer
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import time
import keras_tuner as kt
from keras import regularizers

from fatigue.networks import vectorise_data
from fatigue.neural.helper import preprocess_input


def hmodel(hp, time_input_shape, const_input_shape):
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape, name='time_input')
    const_input = Input(shape=const_input_shape, name='const_input')

    # Feed time_input through Masking and LSTM layers
    hp_units1 = hp.Int('units1', min_value = 8, max_value = 512, step = 32)
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(hp_units1, return_sequences=False)(time_mask)

    # Concatenate the LSTM output with the constant input
    concat_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers
    hp_units2 = hp.Int('units2', min_value = 8, max_value = 512, step = 32)
    dnn = layers.Dense(hp_units2, activation='relu')(concat_vector)
    life_pred = layers.Dense(1)(dnn)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])

    opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=[keras.metrics.RootMeanSquaredError()])

    return model

def hmodel2(hp, time_input_shape, const_input_shape):
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape, name='time_input')
    const_input = Input(shape=const_input_shape, name='const_input')

    # Feed time_input through Masking and LSTM layers
    hp_units1 = hp.Int('units1', min_value = 4, max_value = 512, step = 32)
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(hp_units1, return_sequences=False)(time_mask)

    # Concatenate the LSTM output with the constant input
    concat_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers
    hp_units2 = hp.Int('units2', min_value = 4, max_value = 512, step = 32)
    last_hidden = layers.Dense(hp_units2, activation = 'relu')(concat_vector)
    
    hp_units3 = hp.Int('units3', min_value = 4, max_value = 512, step = 32)
    dnn = layers.Dense(hp_units3, activation='relu')(last_hidden)
    
    life_pred = layers.Dense(1)(dnn)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])

    opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=[keras.metrics.RootMeanSquaredError()])

    return model

def hmodel3(hp, time_input_shape, const_input_shape):
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape, name='time_input')
    const_input = Input(shape=const_input_shape, name='const_input')

    # Feed time_input through Masking and LSTM layers
    hp_units1 = hp.Int('units1', min_value = 4, max_value = 512, step = 8)
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(hp_units1,
                             kernel_regularizer=regularizers.l1_l2(),
                             recurrent_regularizer=regularizers.l2(1e-5),
                             bias_regularizer=regularizers.l2(1e-5))(time_mask)

    # Concatenate the LSTM output with the constant input
    concat_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers
    hp_units2 = hp.Int('units2', min_value = 4, max_value = 512, step = 8)
    last_hidden = layers.Dense(hp_units2, activation = 'relu')(concat_vector)
    
    dnn = layers.Dropout(0.5)(last_hidden)
    
    life_pred = layers.Dense(1)(dnn)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])

    opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=[keras.metrics.RootMeanSquaredError()])

    return model

print('Loading Data...')
Xv, Xc, y = vectorise_data()

# Target Scaling

y = np.log1p(y)

Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test = train_test_split(Xv, Xc, y)

Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
preprocess_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, max(map(len, Xv)))

tuner = kt.Hyperband(lambda x: hmodel3(x, Xv_train.shape[1:], Xc_train.shape[1:]),
                     objective=kt.Objective("val_root_mean_squared_error", direction="min"),
                     max_epochs=10,
                     factor=3)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search({"time_input": Xv_train, "const_input": Xc_train}, y_train.reshape(-1),
             epochs = 50, validation_split = 0.2, callbacks = [stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=3)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the LSTM layer is
{best_hps.get('units1')}, the optimal number of units in the densely-connected hidden
layer is {best_hps.get('units2')}, and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# print(f"""
# The hyperparameter search for a LSTM-DENSE-DENSE-OUT architecture is complete.
# The optimal number of units in the LSTM layer is {best_hps.get('units1')}, the
# optimal number of units in the first densely-connected hidden layer is
# {best_hps.get('units2')}, the optimal number of units in the second densely-connected hidden
# layer is {best_hps.get('units3')}, and the optimal learning rate for the optimizer
# is {best_hps.get('learning_rate')}.
# """)

model = tuner.hypermodel.build(best_hps)
history = model.fit({"time_input": Xv_train, "const_input": Xc_train}, y_train.reshape(-1),
                    epochs=50, validation_split=0.2)

val_rms_per_epoch = history.history['val_root_mean_squared_error']
best_epoch = val_rms_per_epoch.index(max(val_rms_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit({"time_input": Xv_train, "const_input": Xc_train},
               y_train.reshape(-1), epochs=best_epoch, validation_split=0.2)

eval_result = hypermodel.evaluate((Xv_test, Xc_test), y_test.reshape(-1))
print("[test loss, test rms]:", eval_result)

y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
y_pred = scaler_y.inverse_transform(hypermodel.predict((Xv_test, Xc_test)).reshape(-1, 1)).reshape(-1)

y_true, y_pred = map(np.expm1, [y_true, y_pred])

rmse = mean_squared_error(y_true, y_pred)

print("{}: {:.2f}".format(model.metrics_names[1], rmse))

print(abs(y_true-y_pred)/y_true*100)
# np.savez('mdata/' + 'ydata-11-01-22' , y_obs=y_true, y_pred=y_pred)
