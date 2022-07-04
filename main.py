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

# test = fatigue_data.get_data(950)[0]

# test = random.choice(fatigue_data.data)
# test, = fatigue_data.get_test_from_sample('435')

# test_plastic_strain(test)

# test_strain_from_cycles(test)
# print()

# for test in fatigue_data.data:
#     gr.graph_peaks_from_test(test)

# gr.graph_cycles_from_test(test)


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

# train, test = train_idx['best'], test_idx['best']

# Xv_train = Xv[train]
# y_train = y[train]

# Xc_train = Xc.iloc[train]
# Xc_test = Xc.iloc[test]

# Xv_test = Xv[test]
# y_test = y[test]

# Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, scaler_y = \
#         preprocess_multi_input(Xv_train, Xv_test, Xc_train, Xc_test, y_train, y_test, 120)

# X = test_features(fatigue_data.data)

# Xv, y = slatten(cycles = 15)

# Xv_train, Xv_test, y_train, y_test = train_test_split(Xv, y, random_state=30)

#%%
# n = 30

# errors0 = np.zeros((n, 5))
# errors1 = np.zeros((n, 5))

# for i in range(n):
#     random_state = np.random.randint(1000)
#     print(f'Running 15-10 No No {i+1}/{n}...')
#     errors0[i, 0], errors1[i, 0], _ = run_test_fmodel(None, None,  m_lstm_s, 20, 'best', l0 = 15, l1 = 10)
#     print(f'Running m_gru_r2 No {i+1}/{n}...')
#     errors0[i, 1], errors1[i, 1], _ = run_test_fmodel(None, None,  m_lstm_best3, 20, 'best')
#     print(f'Running r15-10 No {i+1}/{n}...')
#     errors0[i, 2], errors1[i, 2], _ =  run_test_fmodel(None, None,  m_lstm_best2, 20, 'best')
#     # print(f'Running 20-20 No {i+1}/{n}...')
#     # errors0[i, 3], errors1[i, 3], _ = run_test_fmodel(None, None, m_lstm_s, 20, 'best', l0 = 20, l1 = 20)
#     print(f'Running  m_lstm_r2 No {i+1}/{n}...')
#     errors0[i, 4], errors1[i, 4], _ =  run_test_fmodel(None, None,  m_lstm_best, 20, 'best')

# print(errors0.mean(axis=0))
# print(errors1.mean(axis=0))

# np.savez('mdata/errors', err0 = errors0, err1 = errors1)

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
# run_xval_model_f(m_lstm_r, ep = 20, save_all = 'ydata-11-02-22-R3', fold='best')
# run_sval_model(s_lstm_deep_r_drop, ep = 40, save = True)
# 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# run_rd_devmodel('d', m_lstm_dev2, 100, 'ydata-11-05-22-D')
run_rd_devmodel('d', m_lstm_dev22, 10, 'ydata-25-06-22-D-2')
# run_rd_devmodel('d', m_lstm_dev3, 100, 'ydata-19-06-22-D-1')

# random_state = np.random.randint(1000)
# # random_state = 11
# run_xval_model(m_lstm_best, ep = 100, rs = random_state, save_ = 'ydata-11-02-22-L2', save_all = 'ydata-11-02-22-L2')

# run_stest_model(None, None, s_lstm_deep_r_drop, epochs=20)

# random_state = np.random.randint(1000)

# # # _, _, history1 = run_test_fmodel('ydata-13-02-22-M1', 'm2.h5', None, 100, 'best')
# _, _, history1 = run_test_fmodel('ydata-10-06-22-M1', None, m_lstm_best, 100, 'best', cycles=4100)
# _, _, history1 = run_test_devmodel('ydata-19-06-22-M3', None, m_lstm_dev3, 100, 'best', cycles=2000)
# # _, _, history2 = run_test_devmodel('ydata-18-04-22-M6', None, m_lstm_dev2, 100, 'best', cycles=4200)
# # # # # # # # _, _, history1 = run_test_fmodel('ydata-16-02-22-M2', None, hyperx3, 500, 'best', cycles = 120)
# # # # # # # _, _, history1 = run_test_fmodel('ydata-22-02-22-M2', 'm4.h5', None, 91, 'best', loss = 'meap', cycles=120)

# gr.validation.plot_history_loss(history1, 'LOSS')
# gr.validation.plot_history_mape(history1, 'MAPE')
# gr.validation.plot_history_rmse(history1, 'RMSE')


# gr.validation.plot_history_loss(history2, 'LOSS')
# gr.validation.plot_history_mape(history2, 'MAPE')
# gr.validation.plot_history_rmse(history2, 'RMSE')

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
# gr.models2.graph_nn_pred_all('mdata/ydata-03-02-22-LSTM-5000.npz', log=False, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-03-02-22-GRU-60.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-02-02-22-D-1000.npz', log=False, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-02-02-22-R-1000.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/elasticNet-60.npz', log=True, v2 = True)
# gr.models2.graph_nn_pred_all('mdata/ydata-11-05-22-R-2500.npz', log=True, v2 = True)

# gr.models2.graph_nn_pred_all('mdata/ydata-13-02-22-D-1.npz', log=False, v2 = True)

# data = 'ydata-03-02-22-LSTM-500'

# for i in [1, 10, 60, 120, 500, 1000, 5000, 10834]:
# #     gr.models2.graph_nn_1_fold('mdata/ydata-11-02-22-R2-%d.npz'%i, log=False)

# data = 'mdata/ydata-13-02-22-D-1000.npz'

# data = 'mdata/ydata-01-03-22-D-1000.npz'


# # # data = 'mdata/ydata-14-02-22-HD-10.npz'

# data = 'mdata/ydata-11-05-22-D-2000.npz'
# # data = 'mdata/ydata-12-06-22-D-2000.npz'



# # data = 'mdata/ydata-16-06-22-R-2-2000.npz'

# data = 'mdata/ydata-19-06-22-M3.npz'

# log = True

# # # gr.models2.graph_nn_2_fold(data, log = log, which = 'train')
# # gr.models2.graph_nn_1_fold(data, log = log, save = 'r1900.pdf')
# # gr.models2.graph_nn_hist_only(data, which = 'all', bins = 20, v2 = False , save = 'r1900hist.pdf')
# # gr.models2.graph_nn_hist(data, log = log, which = 'both')
# # print(gr.models2.get_meap(data, v2 = False))

# # # gr.models2.graph_nn_hist(data, log = log, bins = 15, which = 'both')

# # # gr.models2.graph_nn_hist_only(data, bins = 15, which = 'both')

# print(gr.models2.get_meap(data, which = 'train'))
# print(gr.models2.get_meap(data, which = 'dev'))
# print(gr.models2.get_meap(data, which = 'test'))
# # print(gr.models2.get_meap(data))

# # gr.models2.graph_nn_1_dev(data, log = log, which = 'train')
# gr.models2.graph_nn_1_dev(data, log = log, which = 'dev')
# gr.models2.graph_nn_1_dev(data, log = log, which = 'test')

# for i in [2]:
# i = 2
# data = f'mdata/ydata-12-04-22-M{i}.npz'
# #     gr.models2.graph_nn_1_dev(data, log = log, which = 'all')
# #     save = 'm3dlin.pdf'
# # save = 'm3.pdf'
# gr.models2.graph_nn_12_dev(data, log = log, save = '2000dev.pdf')
# print(gr.models2.get_meap(data))
# print(gr.models2.get_chi(data))

# gr.models2.graph_nn_1_fold(data, log = log, save = '1900rdev.pdf')

# gr.models2.graph_nn_11_dev(data, log = log)
# # # gr.models2.graph_nn_12_dev(data, log = log, save = '1900ddev.pdf')

# print(gr.models2.get_meap(data))
# print(gr.models2.get_chi(data))
