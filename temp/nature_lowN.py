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
    graph_nn_12_dev, graph_nn_22_dev, graph_nn_hist_only, get_meap, get_chi
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, \
    train_test_split, KFold, PredefinedSplit

from tdt import test_idx, dev_idx, train_idx, Data

np.random.seed(10)

cycles = 10838
# cycles = 2800
drop_strain = False
save = ''

cnats = ['full', 'lin', 'slim']
cnat = cnats[0]

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
                   ylabel = 'Linear Coefficients')#, ylim = (-4.5, 4.5))
    
    plt.xticks(rotation=45, ha='center', fontsize = 11)
    plt.ylabel('Linear Coefficients', fontsize = 11)
    
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
for i in [14, 17, 28, 4, 22, 16, 50]:
    arg = {x.columns[i] : x[x.columns[i]]}
    xx = xx.assign(**arg)

if cnat != 'full':
    x = xx

xx = pd.DataFrame()

if cnat == 'slim':
    for i in [13, 11, 1, 5, 12, 2, 0, 9]:
        arg = {x.columns[i] : x[x.columns[i]]}
        xx = xx.assign(**arg)
    x = xx

if drop_strain:
    x = x.drop('strain', axis = 1)

# xx = pd.DataFrame()
# for i in [1, -1]:
#     arg = {x.columns[i] : x[x.columns[i]]}
#     xx = xx.assign(**arg)
# x = xx

y = np.log1p(y) 

all_y_true_train = []
all_y_pred_train = []
all_y_true_dev = []
all_y_pred_dev = []
all_y_true_test = []
all_y_pred_test = []

fold = 'lowN'
train, dev, test = train_idx[fold], dev_idx[fold], test_idx[fold]    

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

params= dict()
 
params['alpha'] =  np.logspace(-5, 5, 1000, endpoint=True)
params['l1_ratio'] = np.arange(0, 1, 0.001)

regressor = ElasticNet()

ps = PredefinedSplit([-1]*22+[0]*11)

model = RandomizedSearchCV(regressor, params, n_iter = 1000, scoring='r2', cv=ps, verbose=0, refit = True)
# model = RandomizedSearchCV(regressor, params, n_iter = 1000, scoring='r2', cv=4, verbose=0, refit = True)
# model.fit(X_train, Y_train)

model.fit(np.concatenate((X_train, X_dev)), np.concatenate((Y_train, Y_dev)))


print('Best Params:')
print(model.best_params_)
model = ElasticNet(**model.best_params_)

# All features {'l1_ratio': 0.012, 'alpha': 0.12708787092020596}
# Linear features only {'l1_ratio': 0.684, 'alpha': 0.0003400411932703706}
# Slim features only {'l1_ratio': 0.684, 'alpha': 0.0003400411932703706}

# params = {'l1_ratio': 0.741, 'alpha': 0.006650018030431118}

# params = {'l1_ratio': 0.5, 'alpha': 0.5}

# model = ElasticNet(**params)
# model = LinearRegression()

model.fit(np.concatenate((X_train, X_dev)), np.concatenate((Y_train, Y_dev)))
# model.fit(X_train, Y_train)

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
    

r = report_coeff(x.columns, model.coef_.squeeze(), model.intercept_)


r_data = {'y_obs_test': np.array(all_y_true_test), 'y_pred_test': np.array(all_y_pred_test),
          'y_obs_train': np.array(all_y_true_train), 'y_pred_train': np.array(all_y_pred_train),
          'y_obs_dev': np.array(all_y_true_dev), 'y_pred_dev': np.array(all_y_pred_dev),}

log = True

graph_nn_11_dev(r_data, log = log, load = False, which = 'all')
graph_nn_22_dev(r_data, log = log, load = False, save = '')
# graph_nn_hist_only(r_data, load = False, save = '')

# graph_nn_12_dev(r_data, log = log, load = False, save = '')

print(get_meap(r_data, load = False, which = 'train'))
print(get_meap(r_data, load = False, which = 'dev'))
print(get_meap(r_data, load = False, which = 'test'))

print(get_meap(r_data, load = False, which = 'all'))

#%%

max_coeff = np.abs(r.coef).max()

important_feats = r[np.abs(r.coef) >= max_coeff*0.1]

r_vals = []
nx, ny = xScaler.transform(x_train), yScaler.transform(y_train)
mx, my = xScaler.transform(x_test), yScaler.transform(y_test)
for i in range(52):

# for feat in important_feats.index:
#     i = x.columns.to_list().index(feat)
    feat = x.columns[i]
    
    r_val = sp.stats.pearsonr(nx[:,i], ny.reshape(-1))[0]
    # print(feat, r_val)
#     fig, ax = plt.subplots(figsize=(4,4))
#     ax.plot(nx[:,i], ny, 'o', markeredgecolor = 'black', markerfacecolor = 'None')
#     plt.show()
    
    r_vals.append((i, abs(r_val), feat))

r_vals.sort(key = lambda x: x[1])
print([r[0] for r in r_vals if r[1] >= 0.6])

# np.random.seed(13)

rm = np.random.choice(list(range(52)), 9, replace = False)

# %%

fig, taxes = plt.subplots(9, 6, figsize=(12,18), sharey = True)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel("Measured $N_f$", fontsize=17)
# plt.xlabel("Pullying Mass (g)", fontsize=13)

ax = [taxes[i][j] for i in range(9) for j in range(6)]

props = dict(boxstyle='round', facecolor='#b3d9ff', alpha=1)
# props = dict(boxstyle='round', facecolor='wheat', alpha=1)

for i, feat in enumerate(r.index):
    j = x.columns.to_list().index(feat)
    # ax[i].set_xlabel(frname(x.columns[j]), labelpad = -10)
    if i >= 48:
        i += 1
        
    if j in [r[0] for r in r_vals if r[1] >= 0.65]:
        ax[i].set_facecolor((51/255, 204/255, 51/255, 0.2))
    # elif i in [6, 10, 12, 21, 22, 23, 30, 32, 34, 36, 47]:
    #     ax[i].set_facecolor('#ffff99')
    else:
        ax[i].set_facecolor((1, 0, 0, 0.2))
    
    
    ax[i].set_xlabel(frname(feat), labelpad = -14, fontsize = 13)
    ax[i].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax[i].plot(nx[:,j], ny, 'o', markeredgecolor = '#0000ff', markerfacecolor = 'None', markeredgewidth = 1.2, label = 'Train')
    ax[i].plot(mx[:,j], my, 'x', markeredgecolor = '#ff33cc', markerfacecolor = 'None', markeredgewidth = 2, label = 'Test')
    ax[i].text(0.5, 0.05, '$\\rho = %.2f$'%sp.stats.pearsonr(nx[:,j],\
            ny.reshape(-1))[0], transform=ax[i].transAxes, va='bottom', ha = 'center', bbox=props, fontsize = 13)
    
for i in [48, 53]:
    ax[i].axis('off')
    
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol = 2, facecolor = 'white', edgecolor = 'none', \
            framealpha = 0, bbox_to_anchor=(0.61, 0.908), fontsize=14)

path = r'D:\INDEX\Notes\Semester_16\MMAN4953\Thesis C\img'
# plt.savefig(os.path.join(path, 'lin_feats.pdf'), bbox_inches = 'tight')
plt.show()

#%%

fig, taxes = plt.subplots(2, 4, figsize=(8,4), sharey = True)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel("Measured $N_f$", fontsize=13, labelpad = -15)
# plt.xlabel("Pullying Mass (g)", fontsize=13)

ax = [taxes[i][j] for i in range(2) for j in range(4)]

for i, j in enumerate([r[0] for r in r_vals if r[1] >= 0.65]):
    if i >= 4:
        box = ax[i].get_position()
        box.x0 = box.x0 + 0.1
        box.x1 = box.x1 + 0.1
        ax[i].set_position(box)
        # i += 1
    # ax[i].set_facecolor('#b3ffb3')
    ax[i].set_xlabel(frname(x.columns[j]), labelpad = -11, fontsize = 12)
    ax[i].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax[i].plot(nx[:,j], ny, 'o', markeredgecolor = '#0000ff', markerfacecolor = 'None', markeredgewidth = 1.2, label = 'Train')
    ax[i].plot(mx[:,j], my, 'x', markeredgecolor = '#ff33cc', markerfacecolor = 'None', markeredgewidth = 2, label = 'Test')
    ax[i].text(0.5, 0.05, '$\\rho = %.2f$'%sp.stats.pearsonr(nx[:,j],\
            ny.reshape(-1))[0], transform=ax[i].transAxes, va='bottom', ha = 'center', bbox=props, fontsize = 13)

for i in [7]:
    ax[i].axis('off')    

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol = 2, facecolor = 'white', edgecolor = 'k', \
            framealpha = 0, bbox_to_anchor=(0.67, 0.99), fontsize=14)

plt.savefig(os.path.join(path, 'lin_feats_lowN.pdf'), bbox_inches = 'tight')
plt.show()

#%%

# fig, taxes = plt.subplots(4, 4, figsize=(8,8), sharey = True)

# fig.add_subplot(111, frameon=False)
# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# plt.ylabel("Measured $N_f$", fontsize=13, labelpad = -15)
# # plt.xlabel("Pullying Mass (g)", fontsize=13)

# ax = [taxes[i][j] for i in range(4) for j in range(4)]

# for i, j in enumerate([r[0] for r in r_vals if r[1] >= 0.65]):
#     if i >= 12:
#         i += 1
#     ax[i].set_facecolor('#b3ffb3')
#     ax[i].set_xlabel(frname(x.columns[j]), labelpad = -11, fontsize = 12)
#     ax[i].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     ax[i].plot(nx[:,j], ny, 'o', markeredgecolor = 'black', markerfacecolor = 'None')
#     ax[i].plot(mx[:,j], my, 'x', markeredgecolor = '#00ccff', markerfacecolor = 'None', markeredgewidth = 2)
#     ax[i].text(0.5, 0.05, '$\\rho = %.2f$'%sp.stats.pearsonr(nx[:,j],\
#             ny.reshape(-1))[0], transform=ax[i].transAxes, va='bottom', ha = 'center', bbox=props, fontsize = 13)

# for i in [12, 15]:
#     ax[i].axis('off')    

# plt.savefig(os.path.join(r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures', 'lfeats2.svg'), bbox_inches = 'tight')
# plt.show()

# %%

# colors = ['#8000ff', '#ff1ac6', '#00b300']

# X = [X_train, X_dev, X_test]
# Y = [Y_train, Y_dev, Y_test]


# x1 = np.concatenate([X_train[:,0], X_dev[:,0], X_test[:,0]])
# x2 = np.concatenate([X_train[:,1], X_dev[:,1], X_test[:,1]])
# y2 = np.concatenate([Y_train, Y_dev, Y_test])

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# xt = np.linspace(-4, 4, 30)
# yt = np.linspace(-4, 4, 30)
# Xt, Yt = np.meshgrid(xt, yt)
# Zt = model.predict(np.concatenate([Xt.reshape(-1,1), Yt.reshape(-1,1)], axis = 1)).reshape(30, 30)

# for i, x in enumerate(x1):
    
#     z = model.predict(np.reshape((x, x2[i]), (1,-1)))
#     ax.plot3D((x, x), (x2[i], x2[i]), (y2[i][0], z[0]), 'r-')

# for i, col in enumerate(colors):
#     ax.scatter(X[i][:,0], X[i][:,1], Y[i], marker = 'o', s = 60, color = col, alpha= 1)

# ax.plot_surface(Xt, Yt, Zt, color='steelblue', alpha = 0.3, edgecolor = 'None')

# ax.set_xlabel(rename['mean_s']+'[max]')
# ax.set_ylabel(rename['strain'])
# ax.set_zlabel('Fatigue Life')

# plt.show()

#%%

# fig, ax = plt.subplots(1, 1, figsize=(4,4))

# bars = [r'C-M \SI{850}{\celsius}', r'C-M @ \SI{950}{\celsius}', 'RLR', 'RNN']

# xpos = np.arange(len(bars))

# ax.bar(xpos, (18.36, 15.05, 18.67, 13.19), color=['steelblue', 'salmon', 'red', 'blue'])

# ax.set_xticks(xpos)
# ax.set_xticklabels(bars)

# ax.set_ylabel('Mean Absolute Percentage Error (\%)')

# ax.grid(ls = ':', axis = 'y', color = 'gray')

# path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'
# plt.savefig(os.path.join(path, 'bar_err.pdf'), bbox_inches = 'tight')

# plt.show()