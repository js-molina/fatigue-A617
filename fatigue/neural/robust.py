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

from ..networks import vectorise_data, single_input_data
from .helper import preprocess_multi_input, preprocess_single_input, preprocess_multi_input_dev
from .arch import load_known_lstm_model, s_lstm_shallow, m_lstm_dev2
import temp.tdt as tdt
from temp.get_folds import test_idx, train_idx

def robustness(load_func = load_known_lstm_model, clen = 60, n_try = 100, tfeats = [], cfeats = []):
    
    tf.keras.backend.clear_session()
    
    Xv, Xc, y = vectorise_data(tfeats=tfeats, cfeats=cfeats)
    
    # Target Scaling
    y = np.log1p(y)
    
    # Fold and score init
    
    all_y_true_train = []
    all_y_pred_train = []
    all_y_true_test = []
    all_y_pred_test = []
    
    for i in range(n_try):
        
        rs = np.random.randint(10000)
        
        fold = KFold(n_splits=4, shuffle=True, random_state = rs) 
        
        for train, test in fold.split(Xv, y):
            
            Xv_train = Xv[train]
            y_train = y[train]
            
            Xc_train = Xc.iloc[train]
            Xc_test = Xc.iloc[test]
            
            Xv_test = Xv[test]
            y_test = y[test]
            
            Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
            preprocess_multi_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, clen)
            
            model = load_func(Xv_train.shape[1:], Xc_train.shape[1:])
        
            model.fit([Xv_train, Xc_train], y_train, epochs=40, batch_size=33, verbose = 0)
            
            y_true1 = scaler_y.inverse_transform(y_test).reshape(-1)
            y_pred1 = scaler_y.inverse_transform(model.predict((Xv_test, Xc_test))).reshape(-1)
            
            y_true1, y_pred1 = map(np.expm1, [y_true1, y_pred1])
            
            all_y_true_test += y_true1.tolist()
            all_y_pred_test += y_pred1.tolist()
            
            y_true2 = scaler_y.inverse_transform(y_train).reshape(-1)
            y_pred2 = scaler_y.inverse_transform(model.predict((Xv_train, Xc_train))).reshape(-1)
            
            y_true2, y_pred2 = map(np.expm1, [y_true2, y_pred2])
            
            all_y_true_train += y_true2.tolist()
            all_y_pred_train += y_pred2.tolist()
    
    return all_y_true_train, all_y_pred_train, all_y_true_test, all_y_pred_test

def determinism(load_func = load_known_lstm_model, clen = 60, n_try = 100, tfeats = [], cfeats = []):
    
    tf.keras.backend.clear_session()
    
    Xv, Xc, y = vectorise_data(tfeats=tfeats, cfeats=cfeats)
    
    # Target Scaling
    y = np.log1p(y)
    
    # Fold and score init
    
    rs = np.random.randint(10000)
    
    all_y_true_train = []
    all_y_pred_train = []
    all_y_true_test = []
    all_y_pred_test = []
    
    for i in range(n_try):
        
        fold = KFold(n_splits=4, shuffle=True, random_state = rs) 
        
        for train, test in fold.split(Xv, y):
            
            Xv_train = Xv[train]
            y_train = y[train]
            
            Xc_train = Xc.iloc[train]
            Xc_test = Xc.iloc[test]
            
            Xv_test = Xv[test]
            y_test = y[test]
            
            Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
            preprocess_multi_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, clen)
            
            model = load_func(Xv_train.shape[1:], Xc_train.shape[1:])
        
            model.fit([Xv_train, Xc_train], y_train, epochs=40, batch_size=33, verbose = 0)
            
            y_true1 = scaler_y.inverse_transform(y_test).reshape(-1)
            y_pred1 = scaler_y.inverse_transform(model.predict((Xv_test, Xc_test))).reshape(-1)
            
            y_true1, y_pred1 = map(np.expm1, [y_true1, y_pred1])
            
            all_y_true_test += y_true1.tolist()
            all_y_pred_test += y_pred1.tolist()
            
            y_true2 = scaler_y.inverse_transform(y_train).reshape(-1)
            y_pred2 = scaler_y.inverse_transform(model.predict((Xv_train, Xc_train))).reshape(-1)
            
            y_true2, y_pred2 = map(np.expm1, [y_true2, y_pred2])
            
            all_y_true_train += y_true2.tolist()
            all_y_pred_train += y_pred2.tolist()
    
        
    return all_y_true_train, all_y_pred_train, all_y_true_test, all_y_pred_test

def determinism_1(load_func = load_known_lstm_model, clen = 120, n_try = 100, tfeats = [], cfeats = [], fold = 'best'):
    
    Xv, Xc, y = vectorise_data(tfeats=tfeats, cfeats=cfeats)
    
    # Target Scaling
    y = np.log1p(y)
    
    # Fold and score init
    
    train, test = train_idx[fold], test_idx[fold]

    all_y_true_train = []
    all_y_pred_train = []
    all_y_true_test = []
    all_y_pred_test = []
    
    for i in range(n_try):
        
        tf.keras.backend.clear_session()
        
        Xv_train = Xv[train]
        y_train = y[train]

        Xc_train = Xc.iloc[train]
        Xc_test = Xc.iloc[test]

        Xv_test = Xv[test]
        y_test = y[test]
        
        Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
        preprocess_multi_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, clen)
        
        model = load_func(Xv_train.shape[1:], Xc_train.shape[1:])
    
        model.fit([Xv_train, Xc_train], y_train, epochs=40, batch_size=33, verbose = 0)
        
        y_true1 = scaler_y.inverse_transform(y_test).reshape(-1)
        y_pred1 = scaler_y.inverse_transform(model.predict((Xv_test, Xc_test))).reshape(-1)
        
        y_true1, y_pred1 = map(np.expm1, [y_true1, y_pred1])
        
        all_y_true_test += y_true1.tolist()
        all_y_pred_test += y_pred1.tolist()
        
        y_true2 = scaler_y.inverse_transform(y_train).reshape(-1)
        y_pred2 = scaler_y.inverse_transform(model.predict((Xv_train, Xc_train))).reshape(-1)
        
        y_true2, y_pred2 = map(np.expm1, [y_true2, y_pred2])
        
        all_y_true_train += y_true2.tolist()
        all_y_pred_train += y_pred2.tolist()
    
        
    return all_y_true_train, all_y_pred_train, all_y_true_test, all_y_pred_test

def determinism_dev(load_func = load_known_lstm_model, clen = 120, n_try = 100, tfeats = [], cfeats = [], fold = 'best'):
    
    Xv, Xc, y = vectorise_data(tfeats=tfeats, cfeats=cfeats)
    
    # Target Scaling
    y = np.log1p(y)
    
    # Fold and score init
    
    train, dev, test = tdt.train_idx[fold], tdt.dev_idx[fold], tdt.test_idx[fold]

    all_y_true_train = []
    all_y_pred_train = []
    all_y_true_dev = []
    all_y_pred_dev = []
    all_y_true_test = []
    all_y_pred_test = []
    
    for i in range(n_try):
        
        tf.keras.backend.clear_session()

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
        preprocess_multi_input_dev(Xv_train, Xv_dev, Xv_test, Xc_train, Xc_dev, Xc_test, y_train, y_dev, y_test, clen)
             
        model = load_func(Xv_train.shape[1:], Xc_train.shape[1:])
    
        model.fit([Xv_train, Xc_train], y_train, epochs=40, batch_size=33, verbose = 0)
        
            # Inverse normalise target data
    y_true0 = scaler_y.inverse_transform(y_train).reshape(-1)
    y_pred0 = scaler_y.inverse_transform(model.predict((Xv_train, Xc_train))).reshape(-1)
    
    y_true0, y_pred0 = map(np.expm1, [y_true0, y_pred0])
       
    y_true1 = scaler_y.inverse_transform(y_dev).reshape(-1)
    y_pred1 = scaler_y.inverse_transform(model.predict((Xv_dev, Xc_dev))).reshape(-1)
    
    y_true1, y_pred1 = map(np.expm1, [y_true1, y_pred1])    
    
    y_true2 = scaler_y.inverse_transform(y_test).reshape(-1)
    y_pred2 = scaler_y.inverse_transform(model.predict((Xv_test, Xc_test))).reshape(-1)
    
    y_true2, y_pred2 = map(np.expm1, [y_true2, y_pred2])
    
    all_y_true_train += y_true0.tolist()
    all_y_pred_train += y_pred0.tolist()
    
    all_y_true_dev += y_true1.tolist()
    all_y_pred_dev += y_pred1.tolist()
    
    all_y_true_test += y_true2.tolist()
    all_y_pred_test += y_pred2.tolist()
    
    return all_y_true_train, all_y_pred_train, all_y_true_dev, all_y_pred_dev, all_y_true_test, all_y_pred_test