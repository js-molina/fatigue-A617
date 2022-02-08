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
from .helper import preprocess_multi_input, preprocess_single_input
from .arch import load_known_lstm_model, s_lstm_shallow, s_lstmconv_deep
from ..graph import chi_ratio
from ..graph.models2 import graph_nn_prediction
from temp.get_folds import test_idx, train_idx

def run_test_model(save_path = None, model_name = None, load_func = load_known_lstm_model, epochs = 40, rand_st = 31,
                   tfeats = [], cfeats = []):

    tf.keras.backend.clear_session()
    start = time.time()
    print("Starting timer...")
    
    Xv, Xc, y = vectorise_data(tfeats=tfeats, cfeats=cfeats)
    
    # Target Scaling
    
    y = np.log1p(y)
    
    Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test = train_test_split(Xv, Xc, y, random_state=rand_st)
    
    Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
    preprocess_multi_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, 120) 
    
    model = load_func(Xv_train.shape[1:], Xc_train.shape[1:])
    
    history = model.fit((Xv_train,  Xc_train), y_train.reshape(-1), epochs=epochs, batch_size=11, verbose = 0,
                        validation_split = 0.2)
    if model_name:
        model.save('models/' + model_name)
    
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
                   tfeats = [], cfeats = []):

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
    preprocess_multi_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, 120) 
    
    model = load_func(Xv_train.shape[1:], Xc_train.shape[1:])
    
    history = model.fit((Xv_train,  Xc_train), y_train.reshape(-1), epochs=epochs, batch_size=11, verbose = 0,
                        validation_split = 0.2)
    if model_name:
        model.save('models/' + model_name)
    
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

    start = time.time()
    print("Starting timer...")
    
    Xv, Xc, y = vectorise_data()
    
    y = np.log1p(y)
    
    Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test = train_test_split(Xv, Xc, y, random_state=rand_st)
    
    Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
    preprocess_multi_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, max(map(len, Xv))) 
    
    model = load_model(f'models/{model_path}')
    
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    y_pred = scaler_y.inverse_transform(model.predict((Xv_test, Xc_test)).reshape(-1, 1)).reshape(-1)
    
    y_true, y_pred = map(np.expm1, [y_true, y_pred])
    
    rmse = mean_squared_error(y_true, y_pred)
    
    print("{}: {:.2f}".format(model.metrics_names[1], rmse))
    
    end = time.time()
    print("Total time: {}".format(end - start))
    
    print(abs(y_true-y_pred)/y_true*100)
    
    if ydata_name:
        graph_nn_prediction(f'mdata/{ydata_name}.npz')
    else:
        ax = plt.gca()
        
        ax.set_xlabel('Predicted $N_f$')
        ax.set_ylabel('Measured $N_f$')
        
        ax.set_ylim(100, 20000)
        ax.set_xlim(100, 20000)
        
        ax.set_aspect('equal')
        
        ax.loglog(y_pred, y_true, 'rx')
        
        ax.plot([100, 20000], [100, 20000], lw = 2, color = 'k')
        ax.fill_between([100, 20000], 100, [100, 20000], color = 'k', alpha = 0.1)
        
    print(chi_ratio(y_pred, y_true))
    
    plt.show()
