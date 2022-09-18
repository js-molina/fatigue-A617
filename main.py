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

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# run_rd_devmodel('d', m_lstm_dev1, 10, 'ydata-15-07-22-D-1')
# run_rd_devmodel('d', s_lstm_dev1, 10, 'ydata-12-07-22-D-1')

# run_rob_dev(s_lstm_dev1, 100, 'ydata-03-08-22-R-1')
# run_rob_dev(s_lstm_dev2, 100, 'ydata-03-08-22-R-2')


#%%

# tf.keras.backend.clear_session()
# _, _, history1 = run_test_devmodel('ydata-21-07-22-M1', s_lstm_dev1, 400, 'best', cycles=1800, callback = True)

# gr.validation.plot_history_loss(history1, 'LOSS')
# gr.validation.plot_history_mape(history1, 'MAPE')
# gr.validation.plot_history_rmse(history1, 'RMSE')


# %%

data = 'mdata/ydata-03-08-22-R-2.npz'

data = 'mdata/ydata-12-07-22-D-2-10838.npz'

log = True

gr.models2.graph_nn_11_dev(data, log = log)
# gr.models2.graph_nn_22_dev(data, log = log, v1 = True)
gr.models2.graph_nn_12_dev(data, log = log)


c, m = gr.models2.graph_nn_1m_dev(data, log = log, ver = True)


print(gr.models2.get_meap(data, which = 'train'))
print(gr.models2.get_meap(data, which = 'dev'))
print(gr.models2.get_meap(data, which = 'test'), '\n')

print(gr.models2.get_meap(data))
