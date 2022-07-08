#%%
# =============================================================================
# Importing Modules
# =============================================================================
import random, os, sys
from re import M
from fatigue.finder import fatigue_data, cycle_path, fd_to_df
from fatigue.tests.properties import test_plastic_strain, test_strain_from_cycles
from fatigue.tests.strain import test_strain_vals
from fatigue.tests.models import test_morrow
from fatigue.tests.models2 import test_morrow2, test_empirical
from fatigue.tests.peaks import *
import fatigue.graph as gr
import fatigue.strain as st
from fatigue.filter import test_filter
from fatigue.networks import *
from fatigue.neural.running import run_xval_model, run_xval_model_f, run_sval_model, run_rd_model, run_rd_devmodel
from fatigue.neural.test import run_test_model, run_test_loading, run_test_fmodel, run_test_devmodel
from fatigue.neural.helper import *
from fatigue.neural.arch import *
from temp.get_folds import test_idx, train_idx, Data

sys.path.append(os.path.dirname(__file__))

#%% 
# =============================================================================
# Plotting Empirical Model Results
# =============================================================================


# test_strain_vals(fatigue_data)
# test_morrow(fatigue_data)
# test_empirical(fatigue_data)


# cp = gr.graph_all_peaks(temp = 950, strain = 0.6)

#%%

# tfeats = ['plastic_d_m', 's_ratio_d_m']
# cfeats = ['rate']
# Xv, Xc, y = vectorise_data(fatigue_data.data)

#%%

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
run_rd_devmodel('d', m_lstm_dev2, 10, 'ydata-06-07-22-D')
# run_rd_devmodel('d', m_lstm_dev22, 10, 'ydata-25-06-22-D-2')
# run_rd_devmodel('d', m_lstm_dev3, 100, 'ydata-19-06-22-D-1')


#%%

# tf.keras.backend.clear_session()
# _, _, history1 = run_test_devmodel('ydata-06-07-22-M1', None, m_lstm_dev2, 100, 'best', cycles=4100)

# gr.validation.plot_history_loss(history1, 'LOSS')
# gr.validation.plot_history_mape(history1, 'MAPE')
# gr.validation.plot_history_rmse(history1, 'RMSE')


# gr.validation.plot_history_loss(history2, 'LOSS')
# gr.validation.plot_history_mape(history2, 'MAPE')
# gr.validation.plot_history_rmse(history2, 'RMSE')

# %%
# # data = 'mdata/ydata-16-06-22-R-2-2000.npz'

# data = 'mdata/ydata-06-07-22-M1.npz'

# log = True

# gr.models2.graph_nn_11_dev(data, log = log)
# gr.models2.graph_nn_22_dev(data, log = log)
# gr.models2.graph_nn_12_dev(data, log = log)

# print(gr.models2.get_meap(data, which = 'train'))
# print(gr.models2.get_meap(data, which = 'dev'))
# print(gr.models2.get_meap(data, which = 'test'), '\n')

# print(gr.models2.get_meap(data))
# print(gr.models2.get_chi(data))
