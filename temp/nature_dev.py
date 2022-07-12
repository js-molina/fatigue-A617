import os, sys

sys.path.append('..')

import numpy as np
import pandas as pd
import scipy as sp
from fatigue.networks import natural

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.rcParams['text.usetex'] = False

import sklearn
from sklearn.model_selection import KFold
from fatigue.graph.models2 import graph_nn_pred_all, graph_nn_11_dev, graph_nn_12_dev, graph_nn_22_dev, graph_nn_hist, get_meap, get_chi
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, KFold, PredefinedSplit

from tdt import test_idx, dev_idx, train_idx, Data

np.random.seed(10)

cycles = 1900
drop_strain = False
save = ''

rename = {'max_s' : r'$\sigma_{\max}$', 'min_s' : r'$\sigma_{\min}$',
              's_ratio' : r'$\sigma_{r}$', 'mean_s' : r'$\sigma_{m}$', 
              'elastic' : r'$\Delta\varepsilon_{el}$', 'plastic' : r'$\Delta\varepsilon_{pl}$',
              'rate' : r'$\dot{\varepsilon}$', 'strain' : 'Strain Range', 'temp' : 'Temperature'}

def frname(col):
    g = col.split('[')
    if g[0] in rename:
        g[0] = rename[g[0]]
    return '['.join(g)


def report_coeff(names, coef, intercept):
    
    fig, ax = plt.subplots(figsize=(12,3))
    
    r = pd.DataFrame({'coef': coef, 'positive': coef >= 0}, index = names)
    r = r.sort_values(by = ['coef'])
    
    cols = r.index.to_list()
    
    ncols = [frname(col) for col in cols]
    
    r.index = ncols
    
    r['coef'].plot(kind = 'bar', color = r['positive'].map({True: 'b', False: 'r'}),
                   ylabel = 'Coefficients', ylim = (-0.25, 0.25))
    
    path = r'D:\INDEX\Notes\Semester_14\MMAN9451\Thesis A\figs'
    path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'
    plt.savefig(os.path.join(path, 'natsel.pdf'), bbox_inches = 'tight')
    
    plt.show()
    return r

Xv0, Xc, y = natural(cycles=cycles, tfeats=[], cfeats = [])

Xv = Xv0[:]

# Xv = [cell.assign(mean_s = np.mean((cell.max_s, cell.min_s), axis = 0)) for cell in Xv]

# Xv = [cell.drop(['max_s', 'min_s', 'elastic'], axis = 1) for cell in Xv]

cols = []

def mm(x, axis = 0):
    return np.max(abs(x), axis = axis)

def logvar(x, axis = 0):
    return np.log(np.var(x, axis = axis))


stats = np.array(['mean', 'std', 'max', 'min', 'median', 'mm', 'ku', 'sk', 'var'])
funs = np.array([np.mean, np.std, np.max, np.min, np.median, mm, \
                 sp.stats.kurtosis, sp.stats.skew, logvar])

idx = [0,1,2,3,4,6,7,8]
stats = stats[idx]; funs = funs[idx]

for s in stats:
    cols += [col + '[%s]'%s for col in Xv[0].columns]

vals = [np.concatenate([fun(cell, axis = 0) for fun in funs]) for cell in Xv]

Xn = pd.DataFrame(vals, columns = cols)

# Getting first cycle to 95% max

max95 = [0.95 * cell.max_s.max() for cell in Xv0]
cmax95 = [Xv0[i].max_s[Xv0[i].max_s>max95[i]].index[0] for i in range(len(Xv0))]

Xn = Xn.assign(c95max = cmax95)

x = pd.concat([Xn, Xc], axis = 1)

xx = pd.DataFrame()
for i in [48, 14, 34, 33, 17, 28, 22, 4, 16, 50]:
    arg = {x.columns[i] : x[x.columns[i]]}
    xx = xx.assign(**arg)

x = xx

if drop_strain:
    x = x.drop('strain', axis = 1)
    
y = np.log1p(y) 

all_y_true_train = []
all_y_pred_train = []
all_y_true_dev = []
all_y_pred_dev = []
all_y_true_test = []
all_y_pred_test = []


train, dev, test = train_idx['best'], dev_idx['best'], test_idx['best']    

x_train, x_dev, x_test = x.iloc[train], x.iloc[dev], x.iloc[test]
y_train, y_dev, y_test = y[train], y[dev], y[test]

xScaler = PowerTransformer()
xScaler.fit(x_train)

yScaler = StandardScaler()
yScaler.fit(y_train)


X_train, X_dev, X_test = map(xScaler.transform, [x_train, x_dev, x_test])
Y_train, Y_dev, Y_test = map(yScaler.transform, [y_train, y_dev, y_test])

# =============================================================================
# Optimising Parameters
# =============================================================================

# params= dict()
 
# params['alpha'] =  np.logspace(-5, 5, 1000, endpoint=True)
# params['l1_ratio'] = np.arange(0, 1, 0.001)

# regressor = ElasticNet()

# ps = PredefinedSplit([-1]*22+[0]*11)

# model = RandomizedSearchCV(regressor, params, n_iter = 1000, scoring='r2', cv=ps, verbose=-1, refit=True)
# # model.fit(X_train, Y_train)

# model.fit(np.concatenate((X_train, X_dev)), np.concatenate((Y_train, Y_dev)))

# print('Best Params:')
# print(model.best_params_)

# All features
# model = ElasticNet(alpha = 0.06820776732865685, l1_ratio=0.005)


# Linear features only
model = ElasticNet(alpha = 1.148312414543511e-05, l1_ratio=0.137, fit_intercept = True)

model.fit(X_train, Y_train)

pred0 = model.predict(X_train).reshape(-1, 1)
Y_obs0, Y_pred0 = map(yScaler.inverse_transform, [Y_train, pred0])
y_obs0, y_pred0 = map(np.expm1, [Y_obs0, Y_pred0])

pred1 = model.predict(X_dev).reshape(-1, 1)
Y_obs1, Y_pred1 = map(yScaler.inverse_transform, [Y_dev, pred1])
y_obs1, y_pred1 = map(np.expm1, [Y_obs1, Y_pred1])

pred2 = model.predict(X_test).reshape(-1, 1)
Y_obs2, Y_pred2 = map(yScaler.inverse_transform, [Y_test, pred2])
y_obs2, y_pred2 = map(np.expm1, [Y_obs2, Y_pred2])

score0 = sklearn.metrics.mean_absolute_percentage_error(y_obs0, y_pred0)
score1 = sklearn.metrics.mean_absolute_percentage_error(y_obs1, y_pred1)
score2 = sklearn.metrics.mean_absolute_percentage_error(y_obs2, y_pred2)

print(f'Final MAPE: {score0:.3f}/{score1:.3f}/{score2:.3f}')

err = abs(y_obs1-y_pred1)/y_obs1*100

all_y_true_train += y_obs0.tolist()
all_y_pred_train += y_pred0.tolist()
all_y_true_dev += y_obs1.tolist()
all_y_pred_dev += y_pred1.tolist()
all_y_true_test += y_obs2.tolist()
all_y_pred_test += y_pred2.tolist()
    

r = report_coeff(x.columns, model.coef_, model.intercept_)


r_data = {'y_obs_test': np.array(all_y_true_test), 'y_pred_test': np.array(all_y_pred_test),
          'y_obs_train': np.array(all_y_true_train), 'y_pred_train': np.array(all_y_pred_train),
          'y_obs_dev': np.array(all_y_true_dev), 'y_pred_dev': np.array(all_y_pred_dev),}

log = True
# graph_nn_1_dev(r_data, log = log, load = False, which = 'train')
# graph_nn_1_dev(r_data, log = log, load = False, which = 'dev')

graph_nn_11_dev(r_data, log = log, load = False, which = 'all')
graph_nn_22_dev(r_data, log = log, load = False)
# graph_nn_hist(r_data, log = True, load = False, which = 'both', save = f'nat_{cycles}.pdf')

graph_nn_12_dev(r_data, log = log, load = False, save = '1900cnat.pdf')

print(get_meap(r_data, load = False, which = 'train'))
print(get_meap(r_data, load = False, which = 'dev'))
print(get_meap(r_data, load = False, which = 'test'))

print(get_meap(r_data, load = False, which = 'all'))
# print(get_chi(r_data, load = False))

#%%

# r_vals = []
# nx, ny = xScaler.transform(x), yScaler.transform(y)
# for i in range(10):

# # for i in [48, 14, 34, 33, 17, 28, 22, 4, 16, 50]:
    
#     r_val = sp.stats.pearsonr(nx[:,i], ny.reshape(-1))[0]
    
#     print(x.columns[i], r_val)
#     fig, ax = plt.subplots(figsize=(4,4))
#     ax.plot(nx[:,i], ny, 'o', markeredgecolor = 'black', markerfacecolor = 'None')
#     plt.show()
    
#     r_vals.append((i, abs(r_val)))

# np.random.seed(13)

# rm = np.random.choice(list(range(52)), 9, replace = False)

# nx, ny = xScaler.transform(x), yScaler.transform(y)

# fig, taxes = plt.subplots(3, 3, figsize=(9,9), sharey = True)

# fig.add_subplot(111, frameon=False)
# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# plt.ylabel("Measured $N_f$", fontsize=13)
# # plt.xlabel("Pullying Mass (g)", fontsize=13)

# ax = [taxes[i][j] for i in range(3) for j in range(3)]

# for i, j in enumerate(rm):
#     # ax[i].set_xlabel(frname(x.columns[j]), labelpad = -10)
#     ax[i].set_xlabel(frname(x.columns[j]) + ', $\\rho = %.2f$'%sp.stats.pearsonr(nx[:,j], ny.reshape(-1))[0], labelpad = -10)
#     ax[i].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     ax[i].plot(nx[:,j], ny, 'o', markeredgecolor = 'black', markerfacecolor = 'None')

# plt.savefig(os.path.join(r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures', 'linear_feats.pdf'), bbox_inches = 'tight')
# plt.show()
