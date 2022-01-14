#%%
# =============================================================================
# Importing Modules
# =============================================================================

import random
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
from fatigue.neural.validation import run_xval_model
from fatigue.neural.test import run_test_model, run_test_loading
from fatigue.neural.helper import hyperx1_lstm_model, hyperx2_lstm_model

#%%

# =============================================================================
# Plotting Empirical Model Results
# =============================================================================


# test_strain_vals(fatigue_data)
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

# # X, y = vectorise_data(fatigue_data.data)


# X = test_features(fatigue_data.data)


#%%

# run_test_model('ydata-12-01-22', None, hyperx2_lstm_model, 30, 1111)


run_xval_model('ydata-14-01-22-2', hyperx2_lstm_model, ep = 25)
# %%

# run_test_loading(None, model_path='test_model.h5', rand_st=31)

# %%

# gr.models2.graph_nn_prediction('mdata/ydata-11-01-22-2.npz')
# gr.models2.graph_nn_prediction('mdata/ydata-11-01-22-3.npz')
# gr.models2.graph_nn_prediction('mdata/ydata2-11-01-22-2.npz', log = True)
# gr.models2.graph_nn_pred_strain('mdata/ydata-13-01-22-1.npz', log=True)
# gr.models2.graph_nn_prediction('mdata/ydata-13-01-22-1.npz', log = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-13-01-22-1.npz', log=True)

# sd = gr.models2.graph_nn_pred_strain('mdata/ydata-13-01-22-1.npz', log=False)

#%%
