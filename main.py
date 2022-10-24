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
from fatigue.neural.running import run_xval_model, run_xval_model_f, run_sval_model, run_rd_model, run_rd_devmodel, \
                                    run_rob_dev
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
models0 = [full1, full2, full3]
models1 = [full4, full5, full6]

for i, model in enumerate(models0):
    run_rd_devmodel('d', model, 10, 'ydata-24-10-22-L%d'%i)

for i, model in enumerate(models1):
    run_rd_devmodel('d', model, 10, 'ydata-24-10-22-L%d'%(i+3))    
    
# run_rd_devmodel('d', s_lstm_dev1, 10, 'ydata-12-07-22-D-1')

# run_rob_dev(s_lstm_dev1, 100, 'ydata-03-08-22-R-1')
# run_rob_dev(s_lstm_dev2, 100, 'ydata-03-08-22-R-2')


#%%

# tf.keras.backend.clear_session()
# # _, _, history1 = run_test_devmodel('ydata-20-09-22-N', s_lstm_dev2, 400, 'lowN', cycles=200, callback = True)
# # _, _, history1 = run_test_devmodel('ydata-05-10-22-M', s_lstm_dev2, 400, 'best', cycles=1000, callback = True)
# _, _, history1 = run_test_devmodel('ydata-11-10-22-F4', full5, 400, 'best', cycles=1000, callback = True)

# gr.validation.plot_history_loss(history1, 'LOSS')
# gr.validation.plot_history_mape(history1, 'MAPE')
# gr.validation.plot_history_rmse(history1, 'RMSE')


# %%

# Cross Validation Results (s_lstm_dev2)
# data = 'mdata/ydata-03-08-22-R-2.npz'

# Deterministic Results (s_lstm_dev2)
# data = 'mdata/ydata-12-07-22-D-2-10838.npz'

# Low-N Results (s_lstm_lowN2)
# data = 'mdata/ydata-20-09-22-N-10838.npz'

# Cross Validation Results (Nature)
# data = 'mdata/ydata-02-08-22.npz'

# data = 'mdata/ydata-20-09-22-N.npz'

# data = 'mdata/ydata-04-10-22-B1.npz'

# # data = 'mdata/ydata-06-10-22-B1.npz'
# data = 'mdata/ydata-07-10-22-T1.npz'
# data = 'mdata/ydata-11-10-22-T2.npz'
# data = 'mdata/ydata-11-10-22-F1.npz'
# data = 'mdata/ydata-05-10-22-M.npz'

# d = np.load(data)
# trials = []
# for i in range(10):
#     r = dict()
#     for file in d.files:
#         r[file] = d[file].reshape(10,-1)[i]
#     trials.append(r)

# meaps = []
# for trial in trials:
#     meaps.append(gr.models2.get_meap(trial, load = False))
    
# # data = trials[meaps.index(min(meaps))]
# data = trials[meaps.index(max(meaps))]

# load = True
# log = True
# v2 = True

# gr.models2.graph_nn_11_dev(data, log = log, load = load, v2 = v2)
# gr.models2.graph_nn_22_dev(data, log = log, load = load)
# # gr.models2.graph_nn_12_dev(data, log = log, load = load, save = '')

# # c, dd = gr.models2.graph_nn_1m_dev(data, log = log, ver = True, load = load, v2 = v2)
# # gr.models2.graph_nn_11_dev(dd, log = log, load = False, v2 = v2)
# # gr.models2.graph_nn_22_dev(dd, log = log, load = False)

# print(gr.models2.get_meap(data, which = 'train', load = load, v2 = v2))
# print(gr.models2.get_meap(data, which = 'dev', load = load))
# print(gr.models2.get_meap(data, which = 'test', load = load, v2 = v2), '\n')

# print(gr.models2.get_meap(data, load = load, v2 = v2))
