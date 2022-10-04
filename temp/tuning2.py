import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)

sys.path.append('..')

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

def nmodel(hp, time_input_shape, const_input_shape, nlayer):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    hp_lstm_kr1 = hp.Float('lstm_kr1', min_value = 1e-12, max_value = 1e-1, sampling = 'log')
    hp_lstm_rr1 = hp.Float('lstm_rr1', min_value = 1e-12, max_value = 1e-1, sampling = 'log')
    hp_lstm_br1 = hp.Float('lstm_br1', min_value = 1e-12, max_value = 1e-1, sampling = 'log')
    hp_lstm_kr2 = hp.Float('lstm_kr2', min_value = 1e-12, max_value = 1e-1, sampling = 'log')
    hp_lstm_rr2 = hp.Float('lstm_rr2', min_value = 1e-12, max_value = 1e-1, sampling = 'log')
    hp_lstm_br2 = hp.Float('lstm_br2', min_value = 1e-12, max_value = 1e-1, sampling = 'log')


    hp_lstm_n = hp.Int('lstm_n', min_value = 16, max_value = 64, sampling = 'linear')

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(hp_lstm_n, kernel_regularizer=regularizers.l1_l2(hp_lstm_kr1, hp_lstm_kr2),
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
    for i in range(nlayer):
        hp_hidden_units.append(hp.Int('hidden_units_%d'%i, min_value = 16, max_value = 64, sampling = 'linear'))
        hp_hidden_kr1.append(hp.Float('hidden_kr1_%d'%i, min_value = 1e-12, max_value = 1e-1, sampling = 'log'))
        hp_hidden_br1.append(hp.Float('hidden_br1_%d'%i, min_value = 1e-12, max_value = 1e-1, sampling = 'log'))
        hp_hidden_kr2.append(hp.Float('hidden_kr2_%d'%i, min_value = 1e-12, max_value = 1e-1, sampling = 'log'))
        hp_hidden_br2.append(hp.Float('hidden_br2_%d'%i, min_value = 1e-12, max_value = 1e-1, sampling = 'log'))
    
    # Feed through Dense layers
    for i in range(nlayer):
        temp_vector = layers.Dense(hp_hidden_units[i], kernel_regularizer=regularizers.l1_l2(hp_hidden_kr1[i], hp_hidden_kr2[i]),
                             bias_regularizer=regularizers.l1_l2(hp_hidden_br1[i], hp_hidden_br2[i]), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics = metrics)

    return model


print('Loading Data...')

tfeats, cfeats = [], []
Xv, Xc, y = vectorise_data(tfeats = tfeats, cfeats = cfeats)

# Target Scaling

y = np.log1p(y)

split = 'best'
train, dev, test = train_idx[split], dev_idx[split], test_idx[split]

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

for nlayer in range(1, 11):
	tf.keras.backend.clear_session()
    
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
    
	tf.random.set_seed(10)
	tuner = kt.Hyperband(lambda x: nmodel(x, Xv_train.shape[1:], Xc_train.shape[1:], nlayer),
						  objective=kt.Objective("val_mean_absolute_percentage_error", direction="min"),
						  max_epochs=151, factor=3, hyperband_iterations=3, directory='Tuners',
						  distribution_strategy=tf.distribute.MirroredStrategy(),
						  project_name='dev_10838_b_%d'%nlayer,
						  overwrite = True)

	stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
	# tuner.search((Xv_train, Xc_train), y_train, epochs = 50, validation_split = 0.2, callbacks = [stop_early])
	tuner.search((Xv_train, Xc_train), y_train, epochs = 200, validation_data = ((Xv_dev, Xc_dev), y_dev), callbacks = [stop_early], batch_size = 33)

	best_hps=tuner.get_best_hyperparameters()[0]

	model = tuner.hypermodel.build(best_hps)
	stop_early_loss = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_percentage_error', patience=100)
	model.fit((Xv_train, Xc_train), y_train, epochs=400, validation_data = ((Xv_dev, Xc_dev), y_dev),
	callbacks = [stop_early_loss], verbose = 0, batch_size = 33)

	model.save('models/best_%d.h5'%nlayer)

	y_true0 = scaler_y.inverse_transform(y_train).reshape(-1)
	y_pred0 = scaler_y.inverse_transform(model.predict((Xv_train, Xc_train))).reshape(-1)
	
	y_true0, y_pred0 = map(np.expm1, [y_true0, y_pred0])
	   
	y_true1 = scaler_y.inverse_transform(y_dev).reshape(-1)
	y_pred1 = scaler_y.inverse_transform(model.predict((Xv_dev, Xc_dev))).reshape(-1)
	
	y_true1, y_pred1 = map(np.expm1, [y_true1, y_pred1])    
	
	y_true2 = scaler_y.inverse_transform(y_test).reshape(-1)
	y_pred2 = scaler_y.inverse_transform(model.predict((Xv_test, Xc_test))).reshape(-1)
	
	y_true2, y_pred2 = map(np.expm1, [y_true2, y_pred2])

	np.savez('../mdata/ydata-01-10-22-B%d'%nlayer, y_obs_train=y_true0, y_pred_train=y_pred0,
		  y_obs_dev=y_true1, y_pred_dev=y_pred1, y_obs_test=y_true2, y_pred_test=y_pred2)
