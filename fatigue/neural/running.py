from tabnanny import verbose
import time
import datetime
import tensorflow as tf
import numpy as np

from ..networks import vectorise_data, single_input_data
from .helper import preprocess_multi_input, preprocess_single_input
from .arch import load_known_lstm_model, hyperx1_lstm_model, s_lstm_shallow
from .validation import cross_val_eval, cross_val_single

# =============================================================================
#     Training Setup
# =============================================================================

FOLDS = 4             # Number of folds for cross validation
BATCH = 11               # Batch size of 1 seems to work. Batch size may need to be >=3 if MULTI_GPU=True
GPUS = tf.config.list_logical_devices('GPU')    # List of GPUs

def run_xval_model(load_func = load_known_lstm_model, ep = 40, save = True):
    
    start = time.time()
    print("Starting timer...")

    Xv, Xc, y = vectorise_data()

    [1, 10, 20, 40, 60, 80, 100, 120, 150, 200, 500, 1000, 5000, 10000, 20000]

    for c_len in [120]:
        t1 = time.time()
        print(f'Training NN with {c_len} cycles...')
        rmse_scores, y_true0, y_pred0, y_true1, y_pred1 = cross_val_eval(Xv,Xc, y, n_epochs=ep,
                n_batch=BATCH, c_len=c_len, n_folds = FOLDS, gpu_list=GPUS, load_func = load_func, ver=1, save=True)
        if save:
            np.savez('mdata/ydata-25-01-22-%d'%c_len , y_obs_train=y_true0, y_pred_train=y_pred0,
                                                    y_obs_test=y_true1, y_pred_test=y_pred1)
        y_true0, y_pred0, y_true1, y_pred1 = map(np.array, [y_true0, y_pred0, y_true1, y_pred1])

        err0 = abs(y_true0-y_pred0)/y_true0*100
        err1 = abs(y_true1-y_pred1)/y_true1*100
        print(f'Avg. Training Error: {np.mean(err0):.2f}')
        print(f'Avg. Test Error: {np.mean(err1):.2f}')
        t2 = time.time()
        print(f'Time taken for {FOLDS} folds with {c_len} cycles was {(t2-t1)/60:.2f} minutes.')
    
    end = time.time()
    print("Total time: {:.2f} minutes".format((end - start)/60))

def run_sval_model(load_func = s_lstm_shallow, ep = 30, save = False):
    
    start = time.time()
    print("Starting timer...")
   
    Xv, y = single_input_data()

    [1, 10, 20, 40, 60, 80, 100, 120, 150, 200, 500, 1000, 5000, 10000, 20000]

    for c_len in [150]:
        t1 = time.time()
        print(f'Training NN with {c_len} cycles...')
        rmse_scores, y_true0, y_pred0, y_true1, y_pred1 = cross_val_single(Xv, y, n_epochs=ep,
                n_batch=BATCH, c_len=c_len, n_folds = FOLDS, gpu_list=GPUS, load_func = load_func, save=True)
        if save:
            np.savez('mdata/ydata-21-01-22-sr-%d'%c_len , y_obs_train=y_true0, y_pred_train=y_pred0,
                                                    y_obs_test=y_true1, y_pred_test=y_pred1)
        y_true0, y_pred0, y_true1, y_pred1 = map(np.array, [y_true0, y_pred0, y_true1, y_pred1])

        err0 = abs(y_true0-y_pred0)/y_true0*100
        err1 = abs(y_true1-y_pred1)/y_true1*100
        print(f'Avg. Training Error: {np.mean(err0):.2f}')
        print(f'Avg. Test Error: {np.mean(err1):.2f}')
        t2 = time.time()
        print(f'Time taken for {FOLDS} folds with {c_len} cycles was {(t2-t1)/60:.2f} minutes.')
    
    end = time.time()
    print("Total time: {:.2f} minutes".format((end - start)/60))