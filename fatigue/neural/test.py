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

from ..networks import vectorise_data, single_input_data
from .helper import preprocess_multi_input, preprocess_single_input, preprocess_multi_input_dev
from .arch import load_known_lstm_model, s_lstm_shallow, s_lstmconv_deep
from ..graph import chi_ratio
from ..graph.models2 import graph_nn_prediction
from temp.get_folds import test_idx, train_idx
import temp.tdt as tdt

def run_test_model(save_path = None, model_name = None, load_func = load_known_lstm_model, epochs = 40, rand_st = 31,
                   tfeats = [], cfeats = []):

    history = None
    
    tf.keras.backend.clear_session()
    start = time.time()
    print("Starting timer...")
    
    Xv, Xc, y = vectorise_data(tfeats=tfeats, cfeats=cfeats)
    
    # Target Scaling
    
    y = np.log1p(y)
    
    Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test = train_test_split(Xv, Xc, y, random_state=rand_st)
    
    Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
    preprocess_multi_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, 120) 
    
    if model_name:
        model = load_model('models/' + model_name)
        opt = tf.keras.optimizers.Adam(learning_rate=0.05)
        model.compile(loss='mean_absolute_percentage_error', optimizer=opt, \
                      metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mean_absolute_percentage_error'])
    else:
        model = load_func(Xv_train.shape[1:], Xc_train.shape[1:])
    history = model.fit((Xv_train,  Xc_train), y_train.reshape(-1), epochs=epochs, verbose = 0,
                    validation_data = ((Xv_test,  Xc_test), y_test), batch_size = 33)
    
    # Inverse normalise target data
    y_true1 = scaler_y.inverse_transform(y_test).reshape(-1)
    y_pred1 = scaler_y.inverse_transform(model.predict((Xv_test, Xc_test))).reshape(-1)

    y_true1, y_pred1 = map(np.expm1, [y_true1, y_pred1])

    err1 = abs(y_true1-y_pred1)/y_true1*100

    rmse1 = mean_squared_error(y_true1, y_pred1)

    y_true2 = scaler_y.inverse_transform(y_train).reshape(-1)
    y_pred2 = scaler_y.inverse_transform(model.predict((Xv_train, Xc_train))).reshape(-1)

    y_true2, y_pred2 = map(np.expm1, [y_true2, y_pred2])

    rmse2 = mean_squared_error(y_true2, y_pred2)
    err2 = abs(y_true2-y_pred2)/y_true2*100
    
    if save_path:
        np.savez('mdata/' + save_path, y_obs_train=y_true2, y_pred_train=y_pred2,
                 y_obs_test=y_true1, y_pred_test=y_pred1)

    print(f"Training Error: {min(err2):.2f}, {np.mean(err2):.2f}, {max(err2):.2f}")
    print(f"Testing Error: {min(err1):.2f}, {np.mean(err1):.2f}, {max(err1):.2f}")
    print("Training - {}: {:.2e}".format(model.metrics_names[1], rmse2))
    print("Testing - {}: {:.2e}".format(model.metrics_names[1], rmse1))
    end = time.time()
    print("Total time: {:.2f} minutes".format((end - start)/60))

    return np.mean(err2), np.mean(err1), history

def run_test_fmodel(save_path = None, model_name = None, load_func = load_known_lstm_model, epochs = 40, fold = 'best',
                   tfeats = [], cfeats = [], l0 = 0, l1 = 0, loss = 'huber', cycles = 120):

    if loss == 'meap':
        lss = 'mean_absolute_percentage_error'
    else:
        lss = 'huber_loss'
    
    history = None
    
    tf.keras.backend.clear_session()
    start = time.time()
    print("Starting timer...")
    
    Xv, Xc, y = vectorise_data(tfeats=tfeats, cfeats=cfeats)
    
    # Target Scaling
    
    y = np.log1p(y)
    
    train, test = train_idx[fold], test_idx[fold]

    Xv_train = Xv[train]
    y_train = y[train]

    Xc_train = Xc.iloc[train]
    Xc_test = Xc.iloc[test]

    Xv_test = Xv[test]
    y_test = y[test]
    
    Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
    preprocess_multi_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, cycles) 
    
    if l0 + l1 > 0:
        model = load_func(Xv_train.shape[1:], Xc_train.shape[1:], l0 = l0, l1 = l1)
    else:
        if model_name:
            model = load_model('models/' + model_name)
            opt = tf.keras.optimizers.Adam(learning_rate=0.05)
            model.compile(loss=lss, optimizer=opt, \
                          metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mean_absolute_percentage_error'])
        else:
            model = load_func(Xv_train.shape[1:], Xc_train.shape[1:])
    
    history = model.fit((Xv_train,  Xc_train), y_train.reshape(-1), epochs=epochs, verbose = 0,
                    validation_data = ((Xv_test,  Xc_test), y_test), batch_size = 33)
        
    # Inverse normalise target data
    y_true1 = scaler_y.inverse_transform(y_test).reshape(-1)
    y_pred1 = scaler_y.inverse_transform(model.predict((Xv_test, Xc_test))).reshape(-1)

    y_true1, y_pred1 = map(np.expm1, [y_true1, y_pred1])

    err1 = abs(y_true1-y_pred1)/y_true1*100

    rmse1 = mean_squared_error(y_true1, y_pred1)

    y_true2 = scaler_y.inverse_transform(y_train).reshape(-1)
    y_pred2 = scaler_y.inverse_transform(model.predict((Xv_train, Xc_train))).reshape(-1)

    y_true2, y_pred2 = map(np.expm1, [y_true2, y_pred2])

    rmse2 = mean_squared_error(y_true2, y_pred2)
    err2 = abs(y_true2-y_pred2)/y_true2*100
    
    if save_path:
        np.savez('mdata/' + save_path, y_obs_train=y_true2, y_pred_train=y_pred2,
                 y_obs_test=y_true1, y_pred_test=y_pred1)

    print(f"Training Error: {min(err2):.2f}, {np.mean(err2):.2f}, {max(err2):.2f}")
    print(f"Testing Error: {min(err1):.2f}, {np.mean(err1):.2f}, {max(err1):.2f}")
    print("Training - {}: {:.2e}".format(model.metrics_names[1], rmse2))
    print("Testing - {}: {:.2e}".format(model.metrics_names[1], rmse1))
    end = time.time()
    print("Total time: {:.2f} minutes".format((end - start)/60))

    return np.mean(err2), np.mean(err1), history


def run_test_devmodel(save_path = None, load_func = load_known_lstm_model, epochs = 40, fold = 'best',
                   tfeats = [], cfeats = [], cycles = 100, callback = None):

    history = None
    
    tf.keras.backend.clear_session()
    start = time.time()
    print("Starting timer...")
    
    Xv, Xc, y = vectorise_data(tfeats=tfeats, cfeats=cfeats)
    
    # Target Scaling
    
    y = np.log1p(y)
    
    train, dev, test = tdt.train_idx[fold], tdt.dev_idx[fold], tdt.test_idx[fold]

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
    preprocess_multi_input_dev(Xv_train, Xv_dev, Xv_test, Xc_train, Xc_dev, Xc_test, y_train, y_dev, y_test, cycles)
    
    model = load_func(Xv_train.shape[1:], Xc_train.shape[1:])
    
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_percentage_error', patience=100)] if callback else None
    # stop_early_rmse = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', patience=20)
    
    history = model.fit((Xv_train,  Xc_train), y_train.reshape(-1), epochs=epochs, verbose = 0,
                    validation_data = ((Xv_dev,  Xc_dev), y_dev), callbacks = callbacks, batch_size = 33)
        
    # Inverse normalise target data
    y_true0 = scaler_y.inverse_transform(y_train).reshape(-1)
    y_pred0 = scaler_y.inverse_transform(model.predict((Xv_train, Xc_train))).reshape(-1)
    
    y_true0, y_pred0 = map(np.expm1, [y_true0, y_pred0])
    
    err0 = abs(y_true0-y_pred0)/y_true0*100
    
    rmse0 = mean_squared_error(y_true0, y_pred0)
    
    y_true1 = scaler_y.inverse_transform(y_dev).reshape(-1)
    y_pred1 = scaler_y.inverse_transform(model.predict((Xv_dev, Xc_dev))).reshape(-1)
    
    y_true1, y_pred1 = map(np.expm1, [y_true1, y_pred1])
    
    err1 = abs(y_true1-y_pred1)/y_true1*100
    
    rmse1 = mean_squared_error(y_true1, y_pred1)
    
    y_true2 = scaler_y.inverse_transform(y_test).reshape(-1)
    y_pred2 = scaler_y.inverse_transform(model.predict((Xv_test, Xc_test))).reshape(-1)
    
    y_true2, y_pred2 = map(np.expm1, [y_true2, y_pred2])
    
    err2 = abs(y_true2-y_pred2)/y_true2*100
    
    rmse2 = mean_squared_error(y_true2, y_pred2)
    
    if save_path:
        np.savez('mdata/' + save_path, y_obs_train=y_true0, y_pred_train=y_pred0,
                 y_obs_dev=y_true1, y_pred_dev=y_pred1, y_obs_test=y_true2, y_pred_test=y_pred2)

    print(f"Training Error: {min(err0):.2f}, {np.mean(err0):.2f}, {max(err0):.2f}")
    print(f"Development Error: {min(err1):.2f}, {np.mean(err1):.2f}, {max(err1):.2f}")
    print(f"Testing Error: {min(err2):.2f}, {np.mean(err2):.2f}, {max(err2):.2f}")
    
    print("Training - {}: {:.2e}".format(model.metrics_names[1], rmse0))
    print("Development - {}: {:.2e}".format(model.metrics_names[1], rmse1))
    print("Testing - {}: {:.2e}".format(model.metrics_names[1], rmse2))
    
    end = time.time()
    print("Total time: {:.2f} minutes".format((end - start)/60))
    print("Epochs: {}".format(history.epoch[-1]+1))

    return np.mean(err2), np.mean(err1), history



def run_stest_model(save_path = None, model_name = None, load_func = s_lstm_shallow, epochs = 40, rand_st = 31):

    tf.keras.backend.clear_session()
    start = time.time()
    print("Starting timer...")
    
    Xv, y = single_input_data()
    
    # Target Scaling
    
    y = np.log1p(y)
    
    Xv_train, Xv_test, y_train, y_test = train_test_split(Xv, y, random_state=rand_st)
    
    Xv_train, Xv_test, y_train, y_test, scaler_y = \
    preprocess_single_input(Xv_train, Xv_test, y_train, y_test, 120) 
    
    model = load_func(Xv_train.shape[1:])
    
    model.fit(Xv_train, y_train, epochs=epochs, batch_size=5, verbose = 0)
        
    if model_name:
        model.save('models/' + model_name)
    
    # Inverse normalise target data
    y_true1 = scaler_y.inverse_transform(y_test).reshape(-1)
    y_pred1 = scaler_y.inverse_transform(model.predict(Xv_test)).reshape(-1)

    y_true1, y_pred1 = map(np.expm1, [y_true1, y_pred1])

    err1 = abs(y_true1-y_pred1)/y_true1*100

    rmse1 = mean_squared_error(y_true1, y_pred1)

    y_true2 = scaler_y.inverse_transform(y_train).reshape(-1)
    y_pred2 = scaler_y.inverse_transform(model.predict(Xv_train)).reshape(-1)

    y_true2, y_pred2 = map(np.expm1, [y_true2, y_pred2])

    rmse2 = mean_squared_error(y_true2, y_pred2)
    err2 = abs(y_true2-y_pred2)/y_true2*100

    print(f"Training Error: {min(err2):.2f}, {np.mean(err2):.2f}, {max(err2):.2f}")
    print(f"Testing Error: {min(err1):.2f}, {np.mean(err1):.2f}, {max(err1):.2f}")
    print("Training - {}: {:.2e}".format(model.metrics_names[1], rmse2))
    print("Testing - {}: {:.2e}".format(model.metrics_names[1], rmse1))
    
    end = time.time()
    print("Total time: {:.2f} minutes".format((end - start)/60))

    return np.mean(err2), np.mean(err1)  

def run_test_loading(ydata_name = None, model_path = None, rand_st = 31):
   pass