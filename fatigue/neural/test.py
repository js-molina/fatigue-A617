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

from ..networks import vectorise_data, ragged_numpy_arr
from .helper import load_known_lstm_model, preprocess_input
from ..graph import chi_ratio
from ..graph.models2 import graph_nn_prediction

def run_test_model(save_path = None, model_name = None, rand_st = 31):

    start = time.time()
    print("Starting timer...")
    
    Xv, Xc, y = vectorise_data()
    
    # Target Scaling
    
    y = np.log1p(y)
    
    Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test = train_test_split(Xv, Xc, y, random_state=rand_st)
    
    Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
    preprocess_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, max(map(len, Xv))) 
    
    model = load_known_lstm_model(Xv_train.shape[1:], Xc_train.shape[1:])
    
    model.fit({"time_input": Xv_train, "const_input": Xc_train}, y_train.reshape(-1), epochs=40, batch_size=5)
        
    if model_name:
        model.save('models/' + model_name)
    
    # Inverse normalise target data
    
    y_true = scaler_y.inverse_transform(y_test).reshape(-1)
    y_pred = scaler_y.inverse_transform(model.predict((Xv_test, Xc_test))).reshape(-1)
    
    # Inverse scale data
    y_true, y_pred = map(np.expm1, [y_true, y_pred])
    
    rmse = mean_squared_error(y_true, y_pred)
    
    print("{}: {:.2f}".format(model.metrics_names[1], rmse))
    
    end = time.time()
    print("Total time: {}".format(end - start))
    
    print(abs(y_true-y_pred)/y_true*100)
    
    if save_path:
        np.savez('mdata/' + save_path , y_obs=y_true, y_pred=y_pred)

def run_test_loading(ydata_name = None, model_path = None, rand_st = 31):

    start = time.time()
    print("Starting timer...")
    
    Xv, Xc, y = vectorise_data()
    
    y = np.log1p(y)
    
    Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test = train_test_split(Xv, Xc, y, random_state=rand_st)
    
    Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
    preprocess_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, max(map(len, Xv))) 
    
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
        
    print(chi_ratio(y_pred, y_true))
    
    plt.show()
