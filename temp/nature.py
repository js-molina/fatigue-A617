import os, sys

sys.path.append('..')

import numpy as np
import pandas as pd
from fatigue.networks import natural

import sklearn
from sklearn.model_selection import KFold
from fatigue.graph.models2 import graph_nn_pred_all
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

cycles = 120
drop_strain = False
save = ''


Xv, Xc, y = natural(cycles=cycles)

cols = []

stats = ['mean', 'std', 'max', 'min', 'median']
funs = [np.mean, np.std, np.max, np.min, np.median]

for s in stats:
    cols += [col + '[%s]'%s for col in Xv[0].columns]

vals = [np.concatenate([fun(cell, axis=0) for fun in funs]) for cell in Xv]

Xn = pd.DataFrame(vals, columns = cols)

x = pd.concat([Xn, Xc], axis = 1)

if drop_strain:
    x = x.drop('strain', axis = 1)
    
y = np.log1p(y) 

all_y_true_train = []
all_y_pred_train = []
all_y_true_test = []
all_y_pred_test = []

fold = KFold(n_splits=4, shuffle=True, random_state = 994) 

for n_fold, (train, test) in enumerate(fold.split(x, y)):

    x_train, x_test = x.iloc[train], x.iloc[test]
    y_train, y_test = y[train], y[test]
    
    xScaler = StandardScaler()
    xScaler.fit(x_train)
    
    yScaler = StandardScaler()
    yScaler.fit(y_train)
    
    X_train, X_test = map(xScaler.transform, [x_train, x_test])
    Y_train, Y_test = map(yScaler.transform, [y_train, y_test])
    
    model = ElasticNet(alpha = 0.1, l1_ratio=0.2)
    
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
    
    if save:
        path = '../mdata/break/' + save + '-%d'%cycles
        os.makedirs(path, exist_ok = True)
        np.savez(os.path.join(path, '%d'%(n_fold+1)), x1 = y_pred1, y1 = y_obs1, x0 = y_pred0, y0 = y_obs0)
    
    all_y_true_test += y_obs1.tolist()
    all_y_pred_test += y_pred1.tolist() 
    all_y_true_train += y_obs0.tolist()
    all_y_pred_train += y_pred0.tolist()
    
if save:
    np.savez('../mdata/' + save + '-%d'%cycles , y_obs_train=all_y_true_train, y_pred_train=all_y_pred_train,
                                        y_obs_test=all_y_true_test, y_pred_test=all_y_pred_test)

r_data = {'y_obs_test': np.array(all_y_true_test),'y_pred_test': np.array(all_y_pred_test)}

graph_nn_pred_all(r_data, log = True, v2 = True, load = False)