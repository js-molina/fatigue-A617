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
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, PowerTransformer
from get_folds import test_idx, train_idx, Data

CYC = [5, 10, 50] + list(range(100, 10900, 100))
drop_strain = False
save = ''

err0 = []; err1= []


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

    if drop_strain:
        x = x.drop('strain', axis = 1)
        
    y = np.log1p(y) 

    all_y_true_train = []
    all_y_pred_train = []
    all_y_true_test = []
    all_y_pred_test = []

    train, test = train_idx['best'], test_idx['best']    

    x_train, x_test = x.iloc[train], x.iloc[test]
    y_train, y_test = y[train], y[test]

    xScaler = PowerTransformer()
    xScaler.fit(x_train)

    yScaler = StandardScaler()
    yScaler.fit(y_train)

    X_train, X_test = map(xScaler.transform, [x_train, x_test])
    Y_train, Y_test = map(yScaler.transform, [y_train, y_test])

    model = ElasticNet(alpha = 0.1, l1_ratio=0.01)

    model.fit(X_train, Y_train)

    pred0 = model.predict(X_train).reshape(-1, 1)
    Y_obs0, Y_pred0 = map(yScaler.inverse_transform, [Y_train, pred0])
    y_obs0, y_pred0 = map(np.expm1, [Y_obs0, Y_pred0])

    pred1 = model.predict(X_test).reshape(-1, 1)
    Y_obs1, Y_pred1 = map(yScaler.inverse_transform, [Y_test, pred1])
    y_obs1, y_pred1 = map(np.expm1, [Y_obs1, Y_pred1])

    score0 = sklearn.metrics.mean_absolute_percentage_error(y_obs0, y_pred0)
    score1 = sklearn.metrics.mean_absolute_percentage_error(y_obs1, y_pred1)

    print(f'Final MAPE: {score0:.3f}/{score1:.3f}')

    err = abs(y_obs1-y_pred1)/y_obs1*100

    # if save:
    #     path = '../mdata/break/' + save + '-%d'%cycles
    #     os.makedirs(path, exist_ok = True)
    #     np.savez(os.path.join(path, '%d'%(n_fold+1)), x1 = y_pred1, y1 = y_obs1, x0 = y_pred0, y0 = y_obs0)
        
    all_y_true_test += y_obs1.tolist()
    all_y_pred_test += y_pred1.tolist() 
    all_y_true_train += y_obs0.tolist()
    all_y_pred_train += y_pred0.tolist()
        
    # if save:
    #     np.savez('../mdata/' + save + '-%d'%cycles , y_obs_train=all_y_true_train, y_pred_train=all_y_pred_train,
    #                                         y_obs_test=all_y_true_test, y_pred_test=all_y_pred_test)

    # r = report_coeff(x.columns, model.coef_, model.intercept_)

    r_data = {'y_obs_test': np.array(all_y_true_test),'y_pred_test': np.array(all_y_pred_test),
              'y_obs_train': np.array(all_y_true_train),'y_pred_train': np.array(all_y_pred_train)}
    
    d = r_data
    x0, y0 = d['y_pred_train'], d['y_obs_train']
    x1, y1 = d['y_pred_test'], d['y_obs_test']
    er0 = abs(y0-x0)/y0*100
    er1 = abs(y1-x1)/y1*100
    err0.append(er0)
    err1.append(er1)

merr0 = np.array(list(map(np.mean, err0))).reshape(-1, 1)
merr1 = np.array(list(map(np.mean, err1))).reshape(-1, 1)

err = pd.DataFrame(np.concatenate((merr0, merr1), axis = 1), columns = ['merr0', 'merr1'])

avg_err = err.rolling(10, center=True, min_periods=1).mean()

#%%

vals = CYC

fig, ax = plt.subplots(1, 1, figsize=(4,4))

ax.set_xlabel('Number of Cycles Utilised')
ax.set_ylabel('MAPE (\%)')

ax.set_ylim(5, 30)

ax.plot(vals, merr0, lw = 0.7, color = 'red', alpha = 0.4)
ax.plot(vals, merr1, lw = 0.7, color = 'blue', alpha = 0.4)

ax.plot(vals, avg_err.merr0, lw = 1.5, color = 'red', label = 'Training Data')
ax.plot(vals, avg_err.merr1, lw = 1.5, color = 'blue', label = 'Test Data')

path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'

ax.legend(framealpha = 1, edgecolor = 'None', loc = 0)

# plt.savefig(os.path.join(path, 'breaking_nat.pdf'), bbox_inches = 'tight')

plt.show()

