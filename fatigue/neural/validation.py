from json import load
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, \
    QuantileTransformer, MaxAbsScaler, PowerTransformer, Normalizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import time

from ..networks import vectorise_data
from .helper import preprocess_multi_input
from .arch import load_known_lstm_model, hyperx1_lstm_model

def cross_val_eval(Xv, Xc, y, n_epochs, n_batch, \
                   n_folds, c_len = 120, gpu_list = None, load_func = load_known_lstm_model, save = False):
    
    # Target Scaling
    y = np.log1p(y)
    
    # Fold and score init
    
    fold = KFold(n_splits=n_folds, shuffle=True, random_state = 5) 
    
    rmse_scores = []
    all_y_true = []
    all_y_pred = []
    
    n_fold = 1
    
    for train, test in fold.split(Xv, y):
        
        Xv_train = Xv[train]
        y_train = y[train]
        
        Xc_train = Xc[train]
        Xc_test = Xc[test]
        
        Xv_test = Xv[test]
        y_test = y[test]
        
        Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
        preprocess_multi_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, c_len)
        
        model = load_func(Xv_train.shape[1:], Xc_train.shape[1:])
    
        print('------------------------------------------------------------------------')
        print(f'Training for fold {n_fold} with {Xv_train.shape[1]} cycles...')
    
        model.fit([Xv_train, Xc_train], y_train, epochs=n_epochs, batch_size=n_batch, verbose=1)
        
        y_true1 = scaler_y.inverse_transform(y_test).reshape(-1)
        y_pred1 = scaler_y.inverse_transform(model.predict((Xv_test, Xc_test))).reshape(-1)
        
        y_true1, y_pred1 = map(np.expm1, [y_true1, y_pred1])
        
        err = abs(y_true1-y_pred1)/y_true1*100
        
        print(min(err), np.mean(err), max(err))
        
        all_y_true += y_true1.tolist()
        all_y_pred += y_pred1.tolist()
        
        rmse1 = mean_squared_error(y_true1, y_pred1)
        
        y_true2 = scaler_y.inverse_transform(y_train).reshape(-1)
        y_pred2 = scaler_y.inverse_transform(model.predict((Xv_train, Xc_train))).reshape(-1)
        
        y_true2, y_pred2 = map(np.expm1, [y_true2, y_pred2])
        
        rmse2 = mean_squared_error(y_true2, y_pred2)
        
        if save:
            path = 'mdata/break/%d'%c_len
            os.makedirs(path, exist_ok = True)
            np.savez(os.path.join(path, '%d'%n_fold), x1 = y_pred1, y1 = y_true1, x0 = y_pred2, y0 = y_true2)
        
        rmse_scores.append(rmse1)
        print("Training - {}: {:.2f}".format(model.metrics_names[1], rmse2))
        print("Testing - {}: {:.2f}".format(model.metrics_names[1], rmse1))
        
        n_fold += 1
        
    return rmse_scores, all_y_true, all_y_pred

def run_xval_model(load_func = load_known_lstm_model, ep = 40):
    
    start = time.time()
    print("Starting timer...")
    
# =============================================================================
#     Training Setup
# =============================================================================

    FOLDS = 9             # Number of folds for cross validation
    EPOCHS = ep             # Epoch size of 20-40 appears to work
    BATCH = 6               # Batch size of 1 seems to work. Batch size may need to be >=3 if MULTI_GPU=True
    GPUS = tf.config.list_logical_devices('GPU')    # List of GPUs
    
    Xv, Xc, y = vectorise_data()

    # [1, 10, 20, 40, 60, 80, 100, 120, 150, 200, 500, 1000, 5000, 10000, 20000]

    for c_len in [1, 120, 5000, max(map(len, Xv))]:
        t1 = time.time()
        rmse_scores, y_true, y_pred = cross_val_eval(Xv,Xc, y, n_epochs=EPOCHS,
                n_batch=BATCH, c_len=c_len, n_folds = FOLDS, gpu_list=GPUS, load_func = load_func)
        np.savez('mdata/ydata-20-01-22-%d'%c_len , y_obs=y_true, y_pred=y_pred)
        y_pred, y_true = map(np.array, [y_pred, y_true])

        err = abs(y_true-y_pred)/y_true*100
        print(min(err), np.mean(err), np.max(err))
        t2 = time.time()
        print(f'Time taken for {FOLDS} folds with {c_len} cycles was {(t2-t1)/60} minutes.')
    
    end = time.time()
    print("Total time: {}".format(end - start))

    