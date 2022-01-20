#%%
# =============================================================================
# Importing Modules
# =============================================================================

import random, os
from fatigue.finder import fatigue_data
from fatigue.finder import cycle_path
from fatigue.tests.properties import test_plastic_strain, test_strain_from_cycles
from fatigue.tests.models import test_morrow
from fatigue.tests.strain import test_strain_vals
from fatigue.tests.models2 import test_morrow2, test_empirical
from fatigue.tests.peaks import *
import fatigue.graph as gr
import fatigue.strain as st
from fatigue.filter import test_filter
from fatigue.networks import vectorise_data
# from fatigue.neural.rnn import run_xval_model
from fatigue.neural.running import run_xval_model, run_sval_model
from fatigue.neural.test import run_test_model, run_test_loading
from fatigue.neural.helper import *
from fatigue.neural.arch import *

#%%

# =============================================================================
# Plotting Empirical Model Results
# =============================================================================


# test_strain_vals(fatigue_data)
# test_morrow2(fatigue_data)
# test_empirical(fatigue_data)


#%%

# test = fatigue_data.get_data(950)[0]

# test = random.choice(fatigue_data.data)
# test, = fatigue_data.get_test_from_sample('435')

# test_plastic_strain(test)

# test_strain_from_cycles(test)
# print()


# for test in fatigue_data.data:
#     gr.graph_peaks_from_test(test)

# gr.graph_peaks_from_test(test)

# test_filter(fatigue_data.get_data(850)[-1], lowest_cov = True)

# for test in fatigue_data.data:
#     gr.graph_peaks_from_test(test)

# r = test_some_data(test)

# Xv, Xc, y = vectorise_data(fatigue_data.data)

# Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test = train_test_split(Xv, Xc, y, random_state=30)

# Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
#         preprocess_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, 500)

# X = test_features(fatigue_data.data)


#%%

# run_test_model('ydata-12-01-22', None, hyperx2_lstm_model, 30, 1111)

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
# run_xval_model(hyperx2_lstm_model, ep = 30, save = True)
run_sval_model(s_lstm_shallow, ep = 30, save = True)
# %%

# run_test_loading(None, model_path='test_model.h5', rand_st=31)

# %%

# gr.models2.graph_nn_prediction('mdata/ydata-15-01-22-1.npz', log = True)
# gr.models2.graph_nn_prediction('mdata/ydata-15-01-22-20.npz', log = True)
# gr.models2.graph_nn_prediction('mdata/ydata-11-01-22-3.npz')
# gr.models2.graph_nn_prediction('mdata/ydata2-11-01-22-2.npz', log = True)
# gr.models2.graph_nn_pred_strain('mdata/ydata-13-01-22-1.npz', log=True)
# gr.models2.graph_nn_prediction('mdata/ydata-13-01-22-1.npz', log = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-13-01-22-1.npz', log=True)

# gr.models2.graph_nn_pred_all('mdata/ydata-15-01-22-1.npz', log=True)
# gr.models2.graph_nn_pred_all('mdata/ydata-15-01-22-120.npz', log=True)
# gr.models2.graph_nn_pred_all('mdata/ydata-15-01-22-300.npz', log=True)
# gr.models2.graph_nn_pred_all('mdata/ydata-21-01-22-5000.npz', log=True, v2 = True)

#%%
