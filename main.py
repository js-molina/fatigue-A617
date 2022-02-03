#%%
# =============================================================================
# Importing Modules
# =============================================================================
import random, os
from fatigue.finder import fatigue_data
from fatigue.finder import cycle_path
from fatigue.tests.properties import test_plastic_strain, test_strain_from_cycles
from fatigue.tests.strain import test_strain_vals
from fatigue.tests.models2 import test_morrow2, test_empirical
from fatigue.tests.peaks import *
import fatigue.graph as gr
import fatigue.strain as st
from fatigue.filter import test_filter
from fatigue.networks import *
from fatigue.neural.running import run_xval_model, run_sval_model, run_rd_model
from fatigue.neural.test import run_test_model, run_test_loading, run_stest_model
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

# Xv, y = single_input_data(fatigue_data.data)
# Xv_train, Xv_test, y_train, y_test = train_test_split(Xv, y, random_state=30)

# Xv_train, Xv_test, y_train, y_test, scaler_y = preprocess_single_input_rand(Xv_train, Xv_test, y_train, y_test, 120, 130)

# tfeats = ['plastic_d_m', 's_ratio_d_m']
# cfeats = ['rate']
# Xv, Xc, y = vectorise_data(fatigue_data.data)
# Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test = train_test_split(Xv, Xc, y, random_state=30)

# Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
#         preprocess_multi_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, 500)

# X = test_features(fatigue_data.data)

# Xv, y = slatten(cycles = 15)

# Xv_train, Xv_test, y_train, y_test = train_test_split(Xv, y, random_state=30)

#%%
n = 200

errors0 = np.zeros((n, 5))
errors1 = np.zeros((n, 5))

for i in range(n):
    random_state = np.random.randint(1000)
    print(f'Running m_gru_r No {i+1}/{n}...')
    errors0[i, 0], errors1[i, 0], _ = run_test_model(None, None, m_gru_r, 20, random_state)
    print(f'Running m_gru_r2 No {i+1}/{n}...')
    errors0[i, 1], errors1[i, 1], _ = run_test_model(None, None, m_gru_r2, 20, random_state)
    print(f'Running hyperx2_lstm_model No {i+1}/{n}...')
    errors0[i, 2], errors1[i, 2], _ =  run_test_model(None, None, hyperx2_lstm_model, 20, random_state)
    print(f'Running m_lstm_r No {i+1}/{n}...')
    errors0[i, 3], errors1[i, 3], _ = run_test_model(None, None, m_lstm_r, 20, random_state)
    print(f'Running  m_lstm_r2 No {i+1}/{n}...')
    errors0[i, 4], errors1[i, 4], _ = run_test_model(None, None, m_lstm_r2, 20, random_state)

print(errors0.mean(axis=0))
print(errors1.mean(axis=0))

np.savez('mdata/errors', err0 = errors0, err1 = errors1)

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# run_xval_model(hyperx2_lstm_model, ep = 20)

# random_state = np.random.randint(1000)
# tfeats = ['plastic_d_m', 's_ratio_m', 's_ratio_d_m', 'min_s_m', 'max_s_m']
# cfeats = ['rate']

# random_state = 994

# run_xval_model(m_lstm_r, ep = 40, tfeats = tfeats, cfeats=cfeats, save_all = 'ydata-01-02-22-sparse', rs = random_state)
# run_xval_model(m_lstm_r, ep = 40, save_all = 'ydata-01-02-22-full', rs = random_state)
# run_xval_model(m_lstm_deep_r_l1l2, ep = 40, save_all = 'ydata-01-02-22-v1', rs = random_state)
# run_xval_model(m_lstm_deep_r_l1l2, ep = 40, tfeats = tfeats, cfeats=cfeats, save_all = 'ydata-01-02-22-v12', rs = random_state)
# run_xval_model(m_lstm_r2, ep = 40, save_all = 'ydata-01-02-22-v3', save_ = 'ydata-01-02-22-v3', rs = random_state)
# run_xval_model(m_lstm_r2, ep = 40, save_all = 'ydata-03-02-22-LSTM2', rs = random_state)
# run_sval_model(s_lstm_deep_r_drop, ep = 40, save = True)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# run_rd_model('d', m_lstm_r, 100, 'ydata-02-02-22-D')
# run_rd_model('r', m_lstm_r, 100, 'ydata-02-02-22-R')

# random_state = np.random.randint(1000)
# random_state = 11
# run_xval_model(m_lstm_gru_r_l1l2, ep = 20, rs = random_state, save = 'ydata-27-01-22-GRU%d-'%random_state)

# run_stest_model(None, None, s_lstm_deep_r_drop, epochs=20)

# random_state = np.random.randint(1000)

# _, _, history1 = run_test_model(None, None, m_gru_r_l1l2, 100, random_state)
# _, _, history2 = run_test_model(None, None, m_lstm_r, 100, random_state)

# gr.validation.plot_history_loss(history1, 'GRU')
# gr.validation.plot_history_loss(history2, 'LSTM')
# gr.validation.plot_history_mape(history1, 'GRU')
# gr.validation.plot_history_mape(history2, 'LSTM')
# gr.validation.plot_history_rmse(history1, 'GRU')
# gr.validation.plot_history_rmse(history2, 'LSTM')


# %%
# gr.models2.graph_nn_prediction('mdata/ydata-24-01-22-ml1l2-120.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-21-01-22-120.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-21-01-22-nr-120.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-21-01-22-mrd-120.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-21-01-22-hx2-120.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-21-01-22-mrl1l2-120.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-24-01-22-ml1l2-120.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-25-01-22-500.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-28-01-22-full-60.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-28-01-22-sparse-60.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-31-01-22-full-60.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-31-01-22-sparse-60.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-01-02-22-v1-120.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-01-02-22-v2-120.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-01-02-22-v3-120.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-01-02-22-full-60.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-01-02-22-sparse-60.npz', log=True, v2 = True)

# gr.models2.graph_nn_pred_all('mdata/ydata-02-02-22-D-60.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-02-02-22-R-60.npz', log=True, v2 = True)
