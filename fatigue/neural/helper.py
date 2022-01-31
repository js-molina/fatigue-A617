from keras.models import Model
import pandas as pd
import numpy as np
import seaborn as sb
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras import layers, Input, optimizers, losses, metrics, regularizers
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, PowerTransformer, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
import time
import datetime
from keras.wrappers.scikit_learn import KerasRegressor

def preprocess_single_input(Xv_train, Xv_test, y_train, y_test, seq_max_len, padding = 'post'):
      
    Xv_train_proc = []
    Xv_test_proc = []
    
    for xt in Xv_train:
        Xv_train_proc.append(xt.iloc[:seq_max_len])
    for xt in Xv_test:
        Xv_test_proc.append(xt.iloc[:seq_max_len])

    tempX = pd.concat(Xv_train_proc).reset_index(drop=True)
    scaler_var = PowerTransformer()
    scaler_var.fit(tempX)
    
    # Scaling and normalising data.
    
    Xv_train_proc = np.array([scaler_var.transform(x) for x in Xv_train_proc], dtype='object')
    Xv_test_proc = np.array([scaler_var.transform(x) for x in Xv_test_proc], dtype= 'object')
    
    # Normalising output data.
    
    scaler_y = StandardScaler()
    scaler_y.fit(y_train)
    
    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)
    
    Xv_train_proc = pad_sequences(Xv_train_proc, maxlen = seq_max_len, padding = padding, value = -999, dtype='float64')
    Xv_test_proc = pad_sequences(Xv_test_proc, maxlen = seq_max_len, padding =  padding, value = -999, dtype='float64')
    
    return Xv_train_proc, Xv_test_proc, y_train, y_test, scaler_y

def preprocess_multi_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, seq_max_len, rm = [], padding = 'post'):
    
    Xv_train_proc = []
    Xv_test_proc = []
    
    for xt in Xv_train:
        Xv_train_proc.append(xt.iloc[:seq_max_len])
    for xt in Xv_test:
        Xv_test_proc.append(xt.iloc[:seq_max_len])
    
    tempX = pd.concat(Xv_train_proc).reset_index(drop=True)
    scaler_var = PowerTransformer()
    scaler_var.fit(tempX)
    
    # Scaling and normalising data.
    
    Xv_train_proc = np.array([scaler_var.transform(x) for x in Xv_train_proc], dtype='object')
    Xv_test_proc = np.array([scaler_var.transform(x) for x in Xv_test_proc], dtype= 'object')
    
    scaler_con = StandardScaler()
    scaler_con.fit(Xc_train)
    
    Xc_train = scaler_con.transform(Xc_train)
    Xc_test = scaler_con.transform(Xc_test)
    
    # Normalising output data.
    
    scaler_y = StandardScaler()
    scaler_y.fit(y_train)
    
    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)
    
    Xv_train_proc = pad_sequences(Xv_train_proc, maxlen = seq_max_len, padding = padding, value = -999, dtype='float64')
    Xv_test_proc = pad_sequences(Xv_test_proc, maxlen = seq_max_len, padding =  padding, value = -999, dtype='float64')
    
    return Xv_train_proc, Xv_test_proc, Xc_train, Xc_test, y_train, y_test, scaler_y