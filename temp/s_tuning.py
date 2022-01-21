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

from fatigue.networks import single_input_data
from fatigue.neural.helper import preprocess_single_input

def s_lstm_deep_r_drop_tuning(hp, time_input_shape):

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-2, 1e-1])
    opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    model = keras.Sequential()

    # Create separate inputs for time series and constants
    model.add(Input(shape=time_input_shape))

    hp_units1 = hp.Int('lstm_units', min_value = 8, max_value = 128, step = 16)

    # Feed time_input through Masking and LSTM layers
    model.add(layers.Masking(mask_value=-999))
    model.add(layers.LSTM(hp_units1))
    
    hp_lstm_drop = hp.Choice('lstm_drop', values=[0.2, 0.4, 0.6, 0.8])
    model.add(layers.Dropout(hp_lstm_drop))

    hp_layers = hp.Int('layers', min_value = 1, max_value = 1)

    hp_units2 = []
    hp_hidden_drop = []

    for i in range(hp_layers):
        hp_units2.append(hp.Int('hidden_units%d'%i, min_value = 8, max_value = 512, step = 16))
        hp_hidden_drop.append(hp.Choice('hidden_drop%d'%i, values=[0.2, 0.4, 0.6, 0.8]))

    # Feed through Dense layers
    for i in range(hp_layers):
        model.add(layers.Dense(hp_units2[i], activation='relu'))
        model.add(layers.Dropout(hp_hidden_drop[i]))

    model.add(layers.Dense(1))

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

print('Loading Data...')
Xv, y = single_input_data()

# Target Scaling

y = np.log1p(y)

Xv_train, Xv_test, y_train, y_test = train_test_split(Xv, y, random_state=10000)

Xv_train, Xv_test, y_train, y_test, scaler_y = \
preprocess_single_input(Xv_train, Xv_test, y_train, y_test, 500)

tuner = kt.Hyperband(lambda x: s_lstm_deep_r_drop_tuning(x, Xv_train.shape[1:]),
                     objective=kt.Objective("val_root_mean_squared_error", direction="min"),
                     max_epochs=40,
                     factor=2, directory='Tuners',
                     project_name='s_lstm_r_drop',
                     overwrite = True)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', patience=5)

tuner.search(Xv_train, y_train.reshape(-1), epochs = 50, validation_split = 0.2, callbacks = [stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=3)[0]

print(f'The hyperparameter search is complete. The optimal number of units in the LSTM layer is',
f"{best_hps.get('lstm_units')} with dropout {best_hps.get('lstm_drop')}, and the optimal number",
f"of layers is {best_hps.get('layers')}")

for i in range(best_hps.get('layers')):
    print(f"The units in the densely-connected {i} is {best_hps.get('hidden_units%d'%i)} with",
    f"dropout rate {best_hps.get('hidden_drop%d'%i)}")
     
print(f"The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.")

model = tuner.hypermodel.build(best_hps)
history = model.fit(Xv_train, y_train, epochs=50, validation_split=0.2)

val_rms_per_epoch = history.history['val_root_mean_squared_error']
best_epoch = val_rms_per_epoch.index(max(val_rms_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(Xv_train, y_train, epochs=best_epoch, validation_split=0.2)

eval_result = hypermodel.evaluate(Xv_test, y_test.reshape(-1))
print("[test loss, test rms]:", eval_result)

y_true1 = scaler_y.inverse_transform(y_test).reshape(-1)
y_pred1 = scaler_y.inverse_transform(hypermodel.predict(Xv_test)).reshape(-1)

y_true1, y_pred1 = map(np.expm1, [y_true1, y_pred1])

err1 = abs(y_true1-y_pred1)/y_true1*100

rmse1 = mean_squared_error(y_true1, y_pred1)

y_true2 = scaler_y.inverse_transform(y_train).reshape(-1)
y_pred2 = scaler_y.inverse_transform(hypermodel.predict(Xv_train)).reshape(-1)

y_true2, y_pred2 = map(np.expm1, [y_true2, y_pred2])

err2 = abs(y_true2-y_pred2)/y_true2*100

rmse2 = mean_squared_error(y_true2, y_pred2)

print(f"Training Error: {min(err2):.2f}, {np.mean(err2):.2f}, {max(err2):.2f}")
print(f"Testing Error: {min(err1):.2f}, {np.mean(err1):.2f}, {max(err1):.2f}")
print("Training - {}: {:.2e}".format(model.metrics_names[1], rmse2))
print("Testing - {}: {:.2e}".format(model.metrics_names[1], rmse1))
# np.savez('mdata/' + 'ydata-11-01-22' , y_obs=y_true, y_pred=y_pred)
