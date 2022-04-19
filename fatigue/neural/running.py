from tabnanny import verbose
import time
import datetime
import tensorflow as tf
import numpy as np

from ..networks import vectorise_data, single_input_data
from .helper import preprocess_multi_input, preprocess_single_input
from .arch import load_known_lstm_model, s_lstm_shallow
from .validation import cross_val_eval, cross_val_single, cross_val_evalf
from .robust import robustness, determinism, determinism_1, determinism_dev

# =============================================================================
#     Training Setup
# =============================================================================

FOLDS = 4             # Number of folds for cross validation
BATCH = 33               # Batch size of 1 seems to work. Batch size may need to be >=3 if MULTI_GPU=True
GPUS = tf.config.list_logical_devices('GPU')    # List of GPUs

def run_xval_model(load_func = load_known_lstm_model, ep = 40, save_all = '', save_ = '',
                   rs = 11, tfeats = [], cfeats = []):
    
    start = time.time()
    print("Starting timer...")

    Xv, Xc, y = vectorise_data(tfeats=tfeats, cfeats=cfeats)

    [1, 10, 20, 40, 60, 80, 100, 120, 150, 200, 500, 1000, 5000, 10000, 20000]

    for c_len in [1, 10, 60, 120, 500, 1000, 5000, max(map(len, Xv))]:
        t1 = time.time()
        print(f'Training NN with {c_len} cycles...')
        rmse_scores, y_true0, y_pred0, y_true1, y_pred1 = cross_val_eval(Xv, Xc, y, n_epochs=ep,
                n_batch=BATCH, c_len=c_len, n_folds = FOLDS, rs = rs, load_func = load_func, ver=0, save_=save_)
        if save_all:
            np.savez('mdata/' + save_all + '-%d'%c_len , y_obs_train=y_true0, y_pred_train=y_pred0,
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

def run_xval_model_f(load_func = load_known_lstm_model, ep = 40, save_all = '', save_ = '',
                   fold = 'best', tfeats = [], cfeats = [], l0 = 0, l1 = 0):
    
    start = time.time()
    print("Starting timer...")

    Xv, Xc, y = vectorise_data(tfeats=tfeats, cfeats=cfeats)

    [1, 10, 20, 40, 60, 80, 100, 120, 150, 200, 500, 1000, 5000, 10000, 20000]

    for c_len in [1, 10, 60, 120, 500, 1000, 5000, max(map(len, Xv))]:
        t1 = time.time()
        print(f'Training NN with {c_len} cycles...')
        rmse_scores, y_true0, y_pred0, y_true1, y_pred1 = cross_val_evalf(Xv, Xc, y, n_epochs=ep,
                n_batch=BATCH, c_len=120, n_folds = FOLDS, fold = fold, load_func = load_func, ver=0, save_=save_, l0 = l0, l1 = l1)
        if save_all:
            np.savez('mdata/' + save_all + '-%d'%c_len , y_obs_train=y_true0, y_pred_train=y_pred0,
                                                    y_obs_test=y_true1, y_pred_test=y_pred1)
        y_true0, y_pred0, y_true1, y_pred1 = map(np.array, [y_true0, y_pred0, y_true1, y_pred1])

        err0 = abs(y_true0-y_pred0)/y_true0*100
        err1 = abs(y_true1-y_pred1)/y_true1*100
        print(f'Avg. Training Error: {np.mean(err0):.2f}')
        print(f'Avg. Test Error: {np.mean(err1):.2f}')
        t2 = time.time()
        print(f'Time taken for {fold} fold with {c_len} cycles was {(t2-t1)/60:.2f} minutes.')
    
    end = time.time()
    print("Total time: {:.2f} minutes".format((end - start)/60))

def run_sval_model(load_func = s_lstm_shallow, ep = 30, save = False):
    
    start = time.time()
    print("Starting timer...")
   
    Xv, y = single_input_data()

    [1, 10, 20, 40, 60, 80, 100, 120, 150, 200, 500, 1000, 5000, 10000, 20000]

    for c_len in [1, 10, 120, 500, 1000, 5000, max(map(len, Xv))]:
        save = False
        if c_len == 120:
            save = True
        t1 = time.time()
        print(f'Training NN with {c_len} cycles...')
        rmse_scores, y_true0, y_pred0, y_true1, y_pred1 = cross_val_single(Xv, y, n_epochs=ep,
                n_batch=BATCH, c_len=c_len, n_folds = FOLDS, gpu_list=GPUS, load_func = load_func, save=save)
        if save:
            np.savez('mdata/ydata-27-01-22-grul1l2-%d'%c_len , y_obs_train=y_true0, y_pred_train=y_pred0,
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
    

def run_rd_model(test = 'r', load_func = load_known_lstm_model, n_try = 100, save_ = '', tfeats = [], cfeats = [], light = False):
    
    start = time.time()
    print("Starting timer...")
    
    if test == 'r':
        print('Testing Robustness...')
    elif test == 'd':
        print('Testing Determinism...')
   
    for c_len in range(7300, 10900, 100):
        if test == 'r':
            y_true0, y_pred0, y_true1, y_pred1 = robustness(load_func, c_len, n_try, tfeats, cfeats)
        elif test == 'd':
            if light:
                y_true0, y_pred0, y_true1, y_pred1 = determinism_1(load_func, c_len, n_try, tfeats, cfeats)
            else:
                y_true0, y_pred0, y_true1, y_pred1 = determinism(load_func, c_len, n_try, tfeats, cfeats)
        if save_:
            np.savez('mdata/' + save_ + '-%d'%c_len , y_obs_train=y_true0, y_pred_train=y_pred0,
                                                    y_obs_test=y_true1, y_pred_test=y_pred1)
        y_true0, y_pred0, y_true1, y_pred1 = map(np.array, [y_true0, y_pred0, y_true1, y_pred1])

        err0 = abs(y_true0-y_pred0)/y_true0*100
        err1 = abs(y_true1-y_pred1)/y_true1*100
        print(f'Avg. Training Error: {np.mean(err0):.2f}')
        print(f'Avg. Test Error: {np.mean(err1):.2f}')

    end = time.time()
    print("Total time: {:.2f} minutes".format((end - start)/60))

def run_rd_devmodel(test = 'r', load_func = load_known_lstm_model, n_try = 100, save_ = '', tfeats = [], cfeats = []):
    
    start = time.time()
    print("Starting timer...")
    
    if test == 'r':
        print('Testing Robustness...')
    elif test == 'd':
        print('Testing Determinism...')
   
    for c_len in [5, 10, 50] + list(range(100, 10900, 100)):
        if test == 'r':
            y_true0, y_pred0, y_true1, y_pred1 = robustness(load_func, c_len, n_try, tfeats, cfeats)
            if save_:
                np.savez('mdata/' + save_ + '-%d'%c_len , y_obs_train=y_true0, y_pred_train=y_pred0,
                                                        y_obs_test=y_true1, y_pred_test=y_pred1)
        elif test == 'd':
            y_true0, y_pred0, y_true1, y_pred1, y_true2, y_pred2 = determinism_dev(load_func, c_len, n_try, tfeats, cfeats)
            if save_:
                np.savez('mdata/' + save_ + '-%d'%c_len , y_obs_train=y_true0, y_pred_train=y_pred0,
                    y_obs_dev=y_true1, y_pred_dev=y_pred1, y_obs_test=y_true2, y_pred_test=y_pred2)

    end = time.time()
    print("Total time: {:.2f} minutes".format((end - start)/60))