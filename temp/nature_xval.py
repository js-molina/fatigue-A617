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
from fatigue.graph.models2 import graph_nn_pred_all, graph_nn_11_dev, \
    graph_nn_12_dev, graph_nn_22_dev, graph_nn_hist, get_meap, get_chi, meape
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, \
    train_test_split, KFold, PredefinedSplit

from tdt import test_idx, dev_idx, train_idx, Data

np.random.seed(11)

cycles = 10838
# cycles = 2800
drop_strain = False
save = ''

cnats = ['full', 'lin', 'slim']
cnat = cnats[2]

rename = {'max_s' : r'$\sigma_{\max}$', 'min_s' : r'$\sigma_{\min}$',
              's_ratio' : r'$\sigma_{r}$', 'mean_s' : r'$\sigma_{m}$', 
              'elastic' : r'$\Delta\varepsilon_{el}$', 'plastic' : r'$\Delta\varepsilon_{pl}$',
              'rate' : r'$\dot{\varepsilon}$', 'strain' : r'$\Delta\varepsilon$\,[range]', 'temp' : 'Temperature'}

def frname(col):
    g = col.split('[')
    if g[0] in rename:
        g[0] = rename[g[0]]
    return '['.join(g)


def report_coeff(names, coef, intercept):
    
    fig, ax = plt.subplots(figsize=(12,3))
    
    r = pd.DataFrame({'coef': coef, 'positive': coef >= 0}, index = names)
    
    r = r.sort_values(by = ['coef'])
    
    ax.plot((-20, 20), (0, 0), 'k-', lw = 0.5)
    
    new_names = r.index
    
    cols = r.index.to_list()
    
    ncols = [frname(col) for col in cols]
    
    r.index = ncols
    
    r['coef'].plot(kind = 'bar', color = r['positive'].map({True: 'b', False: 'r'}),
                   ylabel = 'Linear Coefficients', ylim = (-4.5, 4.5))
    
    plt.xticks(rotation=45, ha='center', fontsize = 13)
    plt.ylabel('Linear Coefficients', fontsize = 13)
    
    path = r'D:\INDEX\Notes\Semester_14\MMAN9451\Thesis A\figs'
    path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'
    # plt.savefig(os.path.join(path, 'natsel2.pdf'), bbox_inches = 'tight')
    
    plt.show()
    
    r.index = new_names
    
    return r

Xv0, Xc, y = natural(cycles=cycles, tfeats=[], cfeats = [])

Xv = Xv0[:]

cols = []

def logvar(x, axis = 0):
    return np.log(np.var(x, axis = axis))

stats = np.array(['mean', 'std', 'max', 'min', 'median', 'ku', 'sk', 'var'])
funs = np.array([np.mean, np.std, np.max, np.min, np.median, \
                 sp.stats.kurtosis, sp.stats.skew, logvar])

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
for i in [48, 30, 33, 34, 14, 36, 40, 39, 17, 28, 22, 16, 4, 50]:
    arg = {x.columns[i] : x[x.columns[i]]}
    xx = xx.assign(**arg)

if cnat != 'full':
    x = xx

xx = pd.DataFrame()
for i in [13, 11, 1, 5, 12, 2, 0, 9]:
    arg = {x.columns[i] : x[x.columns[i]]}
    xx = xx.assign(**arg)

if cnat == 'slim':
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
    
fold = KFold(n_splits=9, shuffle=True, random_state = 11) 

r_data = []

ss0 = []
ss2 = []

for train, test in fold.split(x, y):
    
    x_train, x_test = x.iloc[train], x.iloc[test]
    y_train, y_test = y[train], y[test]
    
    xScaler = PowerTransformer()
    xScaler.fit(x_train)
    
    yScaler = StandardScaler()
    yScaler.fit(y_train)
    
    X_train, X_test = map(xScaler.transform, [x_train, x_test])
    Y_train, Y_test = map(yScaler.transform, [y_train, y_test])

    params = {'l1_ratio': 0.684, 'alpha': 0.0003400411932703706}
    
    model = ElasticNet(**params)
    
    model.fit(X_train, Y_train)

    pred0 = model.predict(X_train).reshape(-1, 1)
    Y_obs0, Y_pred0 = map(yScaler.inverse_transform, [Y_train, pred0])
    y_obs0, y_pred0 = map(np.expm1, [Y_obs0, Y_pred0])
    
    pred2 = model.predict(X_test).reshape(-1, 1)
    Y_obs2, Y_pred2 = map(yScaler.inverse_transform, [Y_test, pred2])
    y_obs2, y_pred2 = map(np.expm1, [Y_obs2, Y_pred2])

    score0 = sklearn.metrics.mean_absolute_percentage_error(y_obs0, y_pred0)
    score2 = sklearn.metrics.mean_absolute_percentage_error(y_obs2, y_pred2)

    ss0.append(score0), ss2.append(score2)

    print(f'Final MAPE: {score0:.3f}/{score2:.3f}')

    r_data.append({'y_obs_test': np.array(y_obs2), 'y_pred_test': np.array(y_pred2),
              'y_obs_train': np.array(y_obs0), 'y_pred_train': np.array(y_pred0)})
    
    all_y_true_train += y_obs0.tolist()
    all_y_pred_train += y_pred0.tolist()
    all_y_true_test += y_obs2.tolist()
    all_y_pred_test += y_pred2.tolist()
 

ss0 = np.array(ss0); ss2 = np.array(ss2)

log = True

fig, taxes = plt.subplots(3, 3, sharex=False, sharey = False, figsize=(10,10))
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel("Observed $N_f$", fontsize=13, labelpad=12)
plt.xlabel("Predicted $N_f$", fontsize=13, labelpad=12)

axes = [taxes[i][j] for i in range(3) for j in range(3)]

colors = ['#8000ff', '#00b300']
markers = ['x', 's']
labels = []

ms = 7

for i, r in enumerate(r_data):
    
    x0, y0, x1, y1 = r['y_pred_train'], r['y_obs_train'], r['y_pred_test'], r['y_obs_test']
    
    axes[i].set_ylim(100, 12000)
    axes[i].set_xlim(100, 12000)
    
    if log:
        axes[i].set_ylim(100, 20000)
        axes[i].set_xlim(100, 20000)
        axes[i].set_yscale('log')
        axes[i].set_xscale('log')
    
    
    axes[i].plot([100, 20000], [100, 20000], lw = 2, color = 'k')
    axes[i].fill_between([100, 20000], 100, [100, 20000], color = 'k', alpha = 0.1)
    
    axes[i].plot([100, 20000], [200, 40000], lw = 1, ls = '--', color = 'gray')
    axes[i].plot([200, 40000], [100, 20000], lw = 1, ls = '--', color = 'gray')
    
    axes[i].plot(x0, y0, color = colors[0], ls = 'none', marker = markers[0], markersize = ms, \
              markerfacecolor = 'None', label = 'Train Data', markeredgewidth = 2)
    axes[i].plot(x1, y1, color = colors[1], ls = 'none', marker = markers[1], markersize = ms, \
              markerfacecolor = 'None', label = 'Test Data', markeredgewidth = 2)
    
    mape = '$\\text{MAPE}_{\\text{train}} = %.1f$'%(meape(x0, y0)) + '\% \n' \
    + '$\\text{MAPE}_{\\text{test}} = %.1f$'%meape(x1, y1) + '\%'
    
    axes[i].text(0.35, 0.95, mape, va ='top', ha='center', transform=axes[i].transAxes,
                 bbox={'facecolor': 'w', 'alpha': 1, 'pad': 5, 'edgecolor' : 'None'}, fontsize=11)
    
    # axes[i].set_title(title, fontsize = 11, ha = 'left')
    axes[i].grid(dashes = (1, 5), color = 'gray', lw = 0.7)

handles, labels = axes[i].get_legend_handles_labels()
lgd = fig.legend(handles, labels, ncol = 2, facecolor = 'white', edgecolor = 'none', \
            framealpha = 0, bbox_to_anchor=(0.65, 0.93), fontsize=12)

path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'
plt.savefig(os.path.join(path, 'folds2.pdf'), bbox_inches = 'tight')
   
plt.show()

print(f'{ss0.mean()*100:.1f}/{ss2.mean()*100:.1f}')
