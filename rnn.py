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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, PowerTransformer, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
import time
import datetime
from keras.wrappers.scikit_learn import KerasRegressor

from fatigue.networks import vectorise_data, ragged_numpy_arr

def load_lstm_model(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape, name='time_input')
    const_input = Input(shape=const_input_shape, name='const_input')

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(10, return_sequences=False)(time_mask)

    # Concatenate the LSTM output with the constant input
    concat_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers
    dnn = layers.Dense(5, activation='relu')(concat_vector)
    life_pred = layers.Dense(1)(dnn)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

def cross_val_eval(Xv, Xc, y, n_epochs=20, n_batch=3, target_scaling=True, n_folds=10, gpu_multi = False, gpu_list = None):
    
    if target_scaling:
        y = np.log1p(y)
    
    fold = KFold(n_splits=n_folds, shuffle=True, random_state=77)
    
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
        
        tempX = pd.concat(Xv_train).reset_index(drop=True)
        
        scaler_var = PowerTransformer()
        scaler_var.fit(tempX)
        
        # Scaling and normalising data.
        
        Xv_train = np.array([scaler_var.transform(x) for x in Xv_train], dtype='object')
        Xv_test = np.array([scaler_var.transform(x) for x in Xv_test], dtype= 'object')
        
        scaler_con = StandardScaler()
        scaler_con.fit(Xc_train)
        
        Xc_train = scaler_con.transform(Xc_train)
        Xc_test = scaler_con.transform(Xc_test)
        
        # Normalising output data.
        
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        scaler_y = StandardScaler()
        scaler_y.fit(y_train)
        
        y_train = scaler_y.transform(y_train)
        y_test = scaler_y.transform(y_test)
        
        max_len = max(map(len, Xv))
        
        Xv_train = pad_sequences(Xv_train, maxlen = max_len, padding='post', value = -999, dtype='float64')
        Xv_test = pad_sequences(Xv_test, maxlen = max_len, padding='post', value = -999, dtype='float64')
        
        if gpu_multi:
            strategy = tf.distribute.MirroredStrategy(gpu_list)
            with strategy.scope():
                model = load_lstm_model(Xv_train.shape[1:], Xc_train.shape[1:])
        else:
            model = load_lstm_model(Xv_train.shape[1:], Xc_train.shape[1:])
    
        print('------------------------------------------------------------------------')
        print(f'Training for fold {n_fold} ...')
    
        model.fit({"time_input": Xv_train, "const_input": Xc_train}, y_train, epochs=n_epochs, batch_size=n_batch)
        
        model.save('models/folds2/m%d.h5'%n_fold)
        
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

if __name__ == "__main__":
    
    start = time.time()
    print("Starting timer...")
    
# =============================================================================
#     Training Setup
# =============================================================================

    FOLDS = 5              # Number of folds for cross validation
    EPOCHS = 40             # Epoch size of 20-40 appears to work
    BATCH = 6               # Batch size of 1 seems to work. Batch size may need to be >=3 if MULTI_GPU=True
    PADDING = True          # True (recommended) for post-padding; False for trunacting to shortest vector
    INPUT_SCALING = True    # True (recommended) for scaling input data; False for raw data
    TARGET_SCALING = True   # True (recommended) for scaling target with ln(x+1); False for unscaled target
    SMOOTHING = True        # True (recommended) for noise reduction; False for raw data
    MULTI_GPU = False       # False for single GPU usage; True to use data parallelisation across GPUs;
    GPUS = tf.config.list_logical_devices('GPU')    # List of GPUs
    DATA_CYC = [1,10]       # Cycles to use as input
    DATA_TEMP = [850]       # Temperature of experiments to model
    
    Xv, Xc, y = vectorise_data()

    target_scaling = True
    
    rmse_scores, y_true, y_pred = cross_val_eval(Xv,Xc, y, n_epochs=EPOCHS, n_batch=BATCH, gpu_list=GPUS, n_folds = FOLDS, gpu_multi=MULTI_GPU)
    
    np.savez('mdata/ydata5', y_obs=y_true, y_pred=y_pred)
    
    end = time.time()
    print("Total time: {}".format(end - start))
