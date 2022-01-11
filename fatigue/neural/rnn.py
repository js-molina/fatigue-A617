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
from .helper import load_known_lstm_model, hyperx1_lstm_model, preprocess_input


def cross_val_eval(Xv, Xc, y, n_epochs, n_batch, n_folds, gpu_multi = False, gpu_list = None, load_func = load_known_lstm_model):
    
    # Target Scaling
    y = np.log1p(y)
    
    # Fold and score init
    
    fold = KFold(n_splits=n_folds, shuffle=True)
    
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
        preprocess_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, max(map(len, Xv)))
        
        if gpu_multi:
            strategy = tf.distribute.MirroredStrategy(gpu_list)
            with strategy.scope():
                model = load_func(Xv_train.shape[1:], Xc_train.shape[1:])
        else:
            model = load_func(Xv_train.shape[1:], Xc_train.shape[1:])
    
        print('------------------------------------------------------------------------')
        print(f'Training for fold {n_fold} ...')
    
        model.fit({"time_input": Xv_train, "const_input": Xc_train}, y_train, epochs=n_epochs, batch_size=n_batch)
        
        # model.save('../models/folds2/m%d.h5'%n_fold)
        
        y_true = scaler_y.inverse_transform(y_test).reshape(-1)
        y_pred = scaler_y.inverse_transform(model.predict((Xv_test, Xc_test))).reshape(-1)
        
        y_true, y_pred = map(np.expm1, [y_true, y_pred])
        
        print(abs(y_true-y_pred)/y_true*100)
        
        all_y_true += y_true.tolist()
        all_y_pred += y_pred.tolist()
        
        rmse = mean_squared_error(y_true, y_pred)
        
        rmse_scores.append(rmse)
        print("{}: {:.2f}".format(model.metrics_names[1], rmse))
        
        n_fold += 1
        
    return rmse_scores, all_y_true, all_y_pred

def run_xval_model(save_path = None, load_func = load_known_lstm_model):
    
    start = time.time()
    print("Starting timer...")
    
# =============================================================================
#     Training Setup
# =============================================================================

    FOLDS = 5              # Number of folds for cross validation
    EPOCHS = 40             # Epoch size of 20-40 appears to work
    BATCH = 6               # Batch size of 1 seems to work. Batch size may need to be >=3 if MULTI_GPU=True
    MULTI_GPU = False       # False for single GPU usage; True to use data parallelisation across GPUs;
    GPUS = tf.config.list_logical_devices('GPU')    # List of GPUs
    
    Xv, Xc, y = vectorise_data()

    rmse_scores, y_true, y_pred = cross_val_eval(Xv,Xc, y, n_epochs=EPOCHS,
            n_batch=BATCH, gpu_list=GPUS, n_folds = FOLDS, gpu_multi=MULTI_GPU, load_func = load_func)
    
    if save_path:
        np.savez('mdata/' + save_path , y_obs=y_true, y_pred=y_pred)
    
    end = time.time()
    print("Total time: {}".format(end - start))
