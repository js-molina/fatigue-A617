import os, sys

sys.path.append('..')

import numpy as np
import pandas as pd
import scipy as sp
from fatigue.networks import natural

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# matplotlib.rcParams['text.usetex'] = False

import sklearn
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, PowerTransformer

from tdt import test_idx, dev_idx, train_idx, Data

#%%

CYC = [5, 10, 50] + list(range(100, 10900, 100))
drop_strain = False
save = ''

err0 = []; err1= []; err2 = []


for cycles in CYC:
    Xv0, Xc, y = natural(cycles=cycles, tfeats=[], cfeats = [])

    Xv = Xv0[:]

    Xv = [cell.assign(mean_s = np.mean((cell.max_s, cell.min_s), axis = 0)) for cell in Xv]

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
    for i in [48, 30, 33, 34, 14, 36, 40, 39, 17, 28, 22, 16, 4, 50]:
        arg = {x.columns[i] : x[x.columns[i]]}
        xx = xx.assign(**arg)
    x = xx
    
    xx = pd.DataFrame()
    for i in [13, 11, 1, 5, 12, 2, 0, 9]:
        arg = {x.columns[i] : x[x.columns[i]]}
        xx = xx.assign(**arg)
    x = xx
        
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
    
    # Full Cycles
    model = ElasticNet(alpha = 0.0003400411932703706, l1_ratio= 0.684, max_iter=10000)
    
    # Best Cycles (1900)
    # model = ElasticNet(alpha = 1.1750871309048075e-05, l1_ratio=0.955, max_iter=10000)
    
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

    # if save:
    #     path = '../mdata/break/' + save + '-%d'%cycles
    #     os.makedirs(path, exist_ok = True)
    #     np.savez(os.path.join(path, '%d'%(n_fold+1)), x1 = y_pred1, y1 = y_obs1, x0 = y_pred0, y0 = y_obs0)
        
    all_y_true_train += y_obs0.tolist()
    all_y_pred_train += y_pred0.tolist()
    all_y_true_dev += y_obs1.tolist()
    all_y_pred_dev += y_pred1.tolist()
    all_y_true_test += y_obs2.tolist()
    all_y_pred_test += y_pred2.tolist()
        
    # if save:
    #     np.savez('../mdata/' + save + '-%d'%cycles , y_obs_train=all_y_true_train, y_pred_train=all_y_pred_train,
    #                                         y_obs_test=all_y_true_test, y_pred_test=all_y_pred_test)

    # r = report_coeff(x.columns, model.coef_, model.intercept_)

    r_data = {'y_obs_test': np.array(all_y_true_test), 'y_pred_test': np.array(all_y_pred_test),
              'y_obs_dev': np.array(all_y_true_dev), 'y_pred_dev': np.array(all_y_pred_dev),
              'y_obs_train': np.array(all_y_true_train), 'y_pred_train': np.array(all_y_pred_train)}
    
    d = r_data
    x0, y0 = d['y_pred_train'], d['y_obs_train']
    x1, y1 = d['y_pred_dev'], d['y_obs_dev']
    x2, y2 = d['y_pred_test'], d['y_obs_test']
    er0 = abs(y0-x0)/y0*100
    er1 = abs(y1-x1)/y1*100
    er2 = abs(y2-x2)/y2*100
    err0.append(er0)
    err1.append(er1)
    err2.append(er2)

merr0 = np.array(list(map(np.mean, err0))).reshape(-1, 1)
merr1 = np.array(list(map(np.mean, err1))).reshape(-1, 1)
merr2 = np.array(list(map(np.mean, err2))).reshape(-1, 1)

err = pd.DataFrame(np.concatenate((merr0, merr1, merr2), axis = 1), columns = ['merr0', 'merr1', 'merr2'])

avg_err = err.rolling(10, center=True, min_periods=1).mean()

#%%

vals = CYC

fig, ax = plt.subplots(1, 1, figsize=(4,4))

ax.set_xlabel('Number of Utilised Cycles [-]')
ax.set_ylabel('Mean Absolute Percentage Error (MAPE) [\%]')

ax.set_ylim(5, 40)
ax.set_xlim(0, 12e3)

ax.xaxis.set_minor_locator(MultipleLocator(1000))

ax.xaxis.set_tick_params(which='minor', bottom=True, direction = 'inout', length = 3)


# ax.set_xscale('log')

msize = 5

# ax.plot(vals, merr0, 'o', alpha = 0.4, markerfacecolor = 'None', markeredgewidth = 1, markersize = ms, 
#         markeredgecolor = 'blue', label = 'MEAP Training Data')
# ax.plot(vals, merr1, 's', alpha = 0.4, markerfacecolor = 'None', markeredgewidth = 1, markersize = ms, 
#         markeredgecolor = 'xkcd:green', label = 'MEAP Development Data')
# ax.plot(vals, merr2, 'D', alpha = 0.4, markerfacecolor = 'None', markeredgewidth = 1, markersize = ms, 
#         markeredgecolor = 'red', label = 'MEAP Testing Data')
# ax.plot(vals, np.mean((merr0, merr0, merr1, merr2), axis = 0), 'x', markeredgewidth = 2, color = 'k', markersize = ms+1, label = 'MEAP Overall')


ax.plot(vals, merr0, 'x', markersize = msize+2, ls = 'None', \
        markerfacecolor = 'None', markeredgecolor = '#8000ff', markeredgewidth = 2, label = 'Train')
ax.plot(vals, merr1, 'o', markersize = msize+2, ls = 'None', \
   markerfacecolor = 'None', markeredgecolor = '#ff1ac6', markeredgewidth = 2, label = 'Dev')
ax.plot(vals, merr2, 's', markersize = msize+2, ls = 'None', \
    markerfacecolor = 'None', markeredgecolor = '#00b300', markeredgewidth = 2, label = 'Test')


# ax.plot(vals, merr0, 'o', markerfacecolor = 'None', markeredgewidth = 1, markersize = ms,
#         markeredgecolor = 'blue', label = 'MEAP Training Data')
# ax.plot(vals, np.mean((merr1, merr2), axis = 0), '.', markeredgewidth = 1,
#         color = 'k', markersize = msize+3, label = 'Dev/\nTest')

path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'

ax.legend(framealpha = 1, edgecolor = 'black', loc = 0)

ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)

plt.savefig(os.path.join(path, 'breaking_nat_dev2.pdf'), bbox_inches = 'tight')

plt.show()

