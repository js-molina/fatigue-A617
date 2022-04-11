#%%
import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)

sys.path.append('..')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
from fatigue.neural.helper import preprocess_multi_input, preprocess_multi_input_dev
from tdt import test_idx, dev_idx, train_idx

metrics = [tf.keras.metrics.RootMeanSquaredError(), 'mean_absolute_percentage_error']

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
    model.compile(loss='huber_loss', optimizer=opt, metrics = metrics)

    return model

def hmodel4(hp, time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    hp_lstm_units = hp.Int('lstm_units', min_value = 4, max_value = 64, sampling = 'linear')
    hp_lstm_kr = hp.Float('lstm_kr', min_value = 1e-6, max_value = 1e-1, sampling = 'log')
    hp_lstm_rr = hp.Float('lstm_rr', min_value = 1e-6, max_value = 1e-1, sampling = 'log')
    hp_lstm_br = hp.Float('lstm_br', min_value = 1e-6, max_value = 1e-1, sampling = 'log')

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(hp_lstm_units, kernel_regularizer=regularizers.l1_l2(hp_lstm_kr),
                             recurrent_regularizer=regularizers.l1_l2(hp_lstm_rr),
                             bias_regularizer=regularizers.l1_l2(hp_lstm_br))(time_mask)
    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    hp_hidden_units = []
    hp_hidden_kr = []
    hp_hidden_br = []
    
    # Initialising regularisers
    for i in range(2):
        hp_hidden_units.append(hp.Int('hidden_units_%d'%i, min_value = 4, max_value = 60, sampling = 'linear'))
        hp_hidden_kr.append(hp.Float('hidden_kr_%d'%i, min_value = 1e-6, max_value = 1e-1, sampling = 'log'))
        hp_hidden_br.append(hp.Float('hidden_br_%d'%i, min_value = 1e-6, max_value = 1e-1, sampling = 'log'))
    
    # Feed through Dense layers
    for i in range(2):
        temp_vector = layers.Dense(hp_hidden_units[i], kernel_regularizer=regularizers.l1_l2(hp_hidden_kr[i]),
                             bias_regularizer=regularizers.l1_l2(hp_hidden_br[i]), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics = metrics)

    return model

def hmodel5(hp, time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    hp_gru_units = hp.Int('gru_units', min_value = 8, max_value = 64, step = 8)
    hp_gru_kr = hp.Float('gru_kr', min_value = 1e-5, max_value = 1e-1, sampling = 'log')
    hp_gru_rr = hp.Float('gru_rr', min_value = 1e-5, max_value = 1e-1, sampling = 'log')
    hp_gru_br = hp.Float('gru_br', min_value = 1e-5, max_value = 1e-1, sampling = 'log')

    # Feed time_input through Masking and GRU layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.GRU(hp_gru_units, kernel_regularizer=regularizers.l1_l2(hp_gru_kr),
                             recurrent_regularizer=regularizers.l1_l2(hp_gru_rr),
                             bias_regularizer=regularizers.l1_l2(hp_gru_br))(time_mask)
    # Concatenate the GRU output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    hp_hidden_units = []
    hp_hidden_kr = []
    hp_hidden_br = []
    
    # Initialising regularisers
    for i in range(1):
        hp_hidden_units.append(hp.Int('hidden_units_%d'%i, min_value = 8, max_value = 128, sampling = 'log'))
        hp_hidden_kr.append(hp.Float('hidden_kr_%d'%i, min_value = 1e-5, max_value = 1e-1, sampling = 'log'))
        hp_hidden_br.append(hp.Float('hidden_br_%d'%i, min_value = 1e-5, max_value = 1e-1, sampling = 'log'))
    
    # Feed through Dense layers
    for i in range(1):
        temp_vector = layers.Dense(hp_hidden_units[i], kernel_regularizer=regularizers.l1_l2(hp_hidden_kr[i]),
                             bias_regularizer=regularizers.l1_l2(hp_hidden_br[i]), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics = metrics)

    return model

def hmodel6(hp, time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    hp_lstm_units = hp.Int('lstm_units', min_value = 8, max_value = 64, sampling = 'linear')
    hp_lstm_kr = hp.Float('lstm_kr', min_value = 1e-12, max_value = 1e-1, sampling = 'log')
    hp_lstm_rr = hp.Float('lstm_rr', min_value = 1e-12, max_value = 1e-1, sampling = 'log')
    hp_lstm_br = hp.Float('lstm_br', min_value = 1e-12, max_value = 1e-1, sampling = 'log')

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(hp_lstm_units, kernel_regularizer=regularizers.l1_l2(hp_lstm_kr),
                             recurrent_regularizer=regularizers.l1_l2(hp_lstm_rr),
                             bias_regularizer=regularizers.l1_l2(hp_lstm_br))(time_mask)
    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    hp_hidden_units = []
    hp_hidden_kr = []
    hp_hidden_br = []
    
    # Initialising regularisers
    for i in range(1):
        hp_hidden_units.append(hp.Int('hidden_units_%d'%i, min_value = 8, max_value = 64, sampling = 'linear'))
        hp_hidden_kr.append(hp.Float('hidden_kr_%d'%i, min_value = 1e-12, max_value = 1e-1, sampling = 'log'))
        hp_hidden_br.append(hp.Float('hidden_br_%d'%i, min_value = 1e-12, max_value = 1e-1, sampling = 'log'))
    
    # Feed through Dense layers
    for i in range(1):
        temp_vector = layers.Dense(hp_hidden_units[i], kernel_regularizer=regularizers.l1_l2(hp_hidden_kr[i]),
                             bias_regularizer=regularizers.l1_l2(hp_hidden_br[i]), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics = metrics)

    return model

def hmodel7(hp, time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    hp_lstm_units = hp.Int('lstm_units', min_value = 8, max_value = 64, sampling = 'linear')
    hp_lstm_kr1 = hp.Float('lstm_kr1', min_value = 1e-12, max_value = 1e-1, sampling = 'log')
    hp_lstm_rr1 = hp.Float('lstm_rr1', min_value = 1e-12, max_value = 1e-1, sampling = 'log')
    hp_lstm_br1 = hp.Float('lstm_br1', min_value = 1e-12, max_value = 1e-1, sampling = 'log')
    hp_lstm_kr2 = hp.Float('lstm_kr2', min_value = 1e-12, max_value = 1e-1, sampling = 'log')
    hp_lstm_rr2 = hp.Float('lstm_rr2', min_value = 1e-12, max_value = 1e-1, sampling = 'log')
    hp_lstm_br2 = hp.Float('lstm_br2', min_value = 1e-12, max_value = 1e-1, sampling = 'log')

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(hp_lstm_units, kernel_regularizer=regularizers.l1_l2(hp_lstm_kr1, hp_lstm_kr2),
                             recurrent_regularizer=regularizers.l1_l2(hp_lstm_rr1, hp_lstm_rr2),
                             bias_regularizer=regularizers.l1_l2(hp_lstm_br1, hp_lstm_br2))(time_mask)
    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    hp_hidden_units = []
    hp_hidden_kr1 = []
    hp_hidden_br1 = []
    hp_hidden_kr2 = []
    hp_hidden_br2 = []

    # Initialising regularisers
    for i in range(1):
        hp_hidden_units.append(hp.Int('hidden_units_%d'%i, min_value = 8, max_value = 64, sampling = 'linear'))
        hp_hidden_kr1.append(hp.Float('hidden_kr1_%d'%i, min_value = 1e-12, max_value = 1e-1, sampling = 'log'))
        hp_hidden_br1.append(hp.Float('hidden_br1_%d'%i, min_value = 1e-12, max_value = 1e-1, sampling = 'log'))
        hp_hidden_kr2.append(hp.Float('hidden_kr2_%d'%i, min_value = 1e-12, max_value = 1e-1, sampling = 'log'))
        hp_hidden_br2.append(hp.Float('hidden_br2_%d'%i, min_value = 1e-12, max_value = 1e-1, sampling = 'log'))
    
    # Feed through Dense layers
    for i in range(1):
        temp_vector = layers.Dense(hp_hidden_units[i], kernel_regularizer=regularizers.l1_l2(hp_hidden_kr1[i], hp_hidden_kr2[i]),
                             bias_regularizer=regularizers.l1_l2(hp_hidden_br1[i], hp_hidden_br2[i]), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics = metrics)

    return model

def nrm(hp, n_lay, time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    hp_lstm_units = hp.Int('lstm_units', min_value = 5, max_value = 25, step = 1)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(hp_lstm_units)(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    hp_hidden_units = []

    # Initialising regularisers
    for i in range(n_lay):
        hp_hidden_units.append(hp.Int('hidden_units_%d'%i, min_value = 5, max_value = 25, step = 1))

    # Feed through Dense layers
    for i in range(n_lay):
        temp_vector = layers.Dense(hp_hidden_units[i], activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics = metrics)

    return model

def rm(hp, time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    hp_lstm_kr = hp.Float('lstm_kr', min_value = 1e-12, max_value = 1e-1, sampling = 'log')
    hp_lstm_rr = hp.Float('lstm_rr', min_value = 1e-12, max_value = 1e-1, sampling = 'log')
    hp_lstm_br = hp.Float('lstm_br', min_value = 1e-12, max_value = 1e-1, sampling = 'log')

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(15, kernel_regularizer=regularizers.l1_l2(hp_lstm_kr),
                             recurrent_regularizer=regularizers.l1_l2(hp_lstm_rr),
                             bias_regularizer=regularizers.l1_l2(hp_lstm_br))(time_mask)
    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    hp_hidden_units = [10]
    hp_hidden_kr = []
    hp_hidden_br = []

    # Initialising regularisers
    for i in range(1):
        hp_hidden_kr.append(hp.Float('hidden_kr_%d'%i, min_value = 1e-12, max_value = 1e-1, sampling = 'log'))
        hp_hidden_br.append(hp.Float('hidden_br_%d'%i, min_value = 1e-12, max_value = 1e-1, sampling = 'log'))
    
    # Feed through Dense layers
    for i in range(1):
        temp_vector = layers.Dense(hp_hidden_units[i], kernel_regularizer=regularizers.l1_l2(hp_hidden_kr[i]),
                             bias_regularizer=regularizers.l1_l2(hp_hidden_br[i]), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics = metrics)

    return model

print('Loading Data...')
tfeats = ['plastic_d_m', 's_ratio_m', 's_ratio_d_m', 'min_s_m', 'max_s_m']
cfeats = ['rate']

tfeats, cfeats = [], []
Xv, Xc, y = vectorise_data(tfeats = tfeats, cfeats = cfeats)

# Target Scaling

y = np.log1p(y)

train, dev, test = train_idx['best'], dev_idx['best'], test_idx['best']

Xv_train = Xv[train]
Xv_dev = Xv[dev]
Xv_test = Xv[test]

Xc_train = Xc.iloc[train]
Xc_dev = Xc.iloc[dev]
Xc_test = Xc.iloc[test]

y_train = y[train]
y_dev = y[dev]
y_test = y[test]

Xv_train, Xv_dev, Xv_test, Xc_train, Xc_dev, Xc_test, y_train, y_dev, y_test, scaler_y = \
preprocess_multi_input_dev(Xv_train, Xv_dev, Xv_test, Xc_train, Xc_dev, Xc_test, y_train, y_dev, y_test, 100)

tuner = kt.Hyperband(lambda x: hmodel7(x, Xv_train.shape[1:], Xc_train.shape[1:]),
                      objective=kt.Objective("val_mean_absolute_percentage_error", direction="min"),
                      max_epochs=150, factor=3, hyperband_iterations=1, directory='Tuners',
                      project_name='m_lstm_r5',
                      overwrite = False)

# tuner = kt.BayesianOptimization(lambda x: nrm(x, nr_lay, Xv_train.shape[1:], Xc_train.shape[1:]),
#                      objective=kt.Objective("mean_absolute_percentage_error", direction="min"),
#                      directory='Tuners', project_name='m_lstm_nr1',
#                      overwrite = True)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# tuner.search((Xv_train, Xc_train), y_train, epochs = 50, validation_split = 0.2, callbacks = [stop_early])
tuner.search((Xv_train, Xc_train), y_train, epochs = 200, validation_data = ((Xv_dev, Xc_dev), y_dev), callbacks = [stop_early], batch_size = 33)

best_hps=tuner.get_best_hyperparameters()[0]

for key, val in best_hps.values.items():
    print(key + f' = {val}')

model = tuner.hypermodel.build(best_hps)
history = model.fit((Xv_train, Xc_train), y_train, epochs=200, validation_data = ((Xv_dev, Xc_dev), y_dev), verbose = 0, batch_size = 33)

val_meap_per_epoch = history.history['val_mean_absolute_percentage_error']
val_rmse_per_epoch = history.history['val_root_mean_squared_error']
best_epoch = val_meap_per_epoch.index(min(val_meap_per_epoch)) + 1
# best_epoch = val_rmse_per_epoch.index(min(val_rmse_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit((Xv_train, Xc_train), y_train, epochs=best_epoch, validation_data = ((Xv_dev, Xc_dev), y_test), verbose = 0, batch_size = 33)

hypermodel.save('models/xyz.h5')

eval_result = hypermodel.evaluate((Xv_test, Xc_test), y_test)
print("[test loss, test rms, test mape]:", eval_result)

y_true0 = scaler_y.inverse_transform(y_train).reshape(-1)
y_pred0 = scaler_y.inverse_transform(hypermodel.predict((Xv_train, Xc_train))).reshape(-1)

y_true0, y_pred0 = map(np.expm1, [y_true0, y_pred0])

err0 = abs(y_true0-y_pred0)/y_true0*100

rmse0 = mean_squared_error(y_true0, y_pred0)


y_true1 = scaler_y.inverse_transform(y_dev).reshape(-1)
y_pred1 = scaler_y.inverse_transform(hypermodel.predict((Xv_dev, Xc_dev))).reshape(-1)

y_true1, y_pred1 = map(np.expm1, [y_true1, y_pred1])

err1 = abs(y_true1-y_pred1)/y_true1*100

rmse1 = mean_squared_error(y_true1, y_pred1)


y_true2 = scaler_y.inverse_transform(y_test).reshape(-1)
y_pred2 = scaler_y.inverse_transform(hypermodel.predict((Xv_test, Xc_test))).reshape(-1)

y_true2, y_pred2 = map(np.expm1, [y_true2, y_pred2])

err2 = abs(y_true2-y_pred2)/y_true2*100

rmse2 = mean_squared_error(y_true2, y_pred2)


print(f"Training Error: {min(err0):.2f}, {np.mean(err0):.2f}, {max(err0):.2f}")
print(f"Development Error: {min(err1):.2f}, {np.mean(err1):.2f}, {max(err1):.2f}")
print(f"Testing Error: {min(err2):.2f}, {np.mean(err2):.2f}, {max(err2):.2f}")

print("Training - {}: {:.2e}".format(model.metrics_names[1], rmse0))
print("Development - {}: {:.2e}".format(model.metrics_names[1], rmse1))
print("Testing - {}: {:.2e}".format(model.metrics_names[1], rmse2))

