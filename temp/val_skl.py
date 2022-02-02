import os, sys

sys.path.append('..')

import numpy as np
from fatigue.networks import slatten
import sklearn
from sklearn.model_selection import KFold
from fatigue.graph.models2 import graph_nn_pred_all
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

c_len = 60

fold = KFold(n_splits=4, shuffle=True, random_state = 994) 

x, y = slatten(cycles = c_len)

y = np.log1p(y)

all_y_true_train = []
all_y_pred_train = []
all_y_true_test = []
all_y_pred_test = []

save = ''

for n_fold, (train, test) in enumerate(fold.split(x, y)):
    
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]
    
    xScaler = StandardScaler()
    xScaler.fit(x_train)

    yScaler = StandardScaler()
    yScaler.fit(y_train)

    X_train, X_test = map(xScaler.transform, [x_train, x_test])
    Y_train, Y_test = map(yScaler.transform, [y_train, y_test])

    model = ElasticNet(alpha = 0.1)
    
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
    
    if save:
        path = '../mdata/break/' + save + '-%d'%c_len
        os.makedirs(path, exist_ok = True)
        np.savez(os.path.join(path, '%d'%(n_fold+1)), x1 = y_pred1, y1 = y_obs1, x0 = y_pred0, y0 = y_obs0)

    all_y_true_test += y_obs1.tolist()
    all_y_pred_test += y_pred1.tolist() 
    all_y_true_train += y_obs0.tolist()
    all_y_pred_train += y_pred0.tolist()

if save:
    np.savez('../mdata/' + save + '-%d'%c_len , y_obs_train=all_y_true_train, y_pred_train=all_y_pred_train,
                                        y_obs_test=all_y_true_test, y_pred_test=all_y_pred_test)

r_data = {'y_obs_test': np.array(all_y_true_test),'y_pred_test': np.array(all_y_pred_test)}
graph_nn_pred_all(r_data, log = True, v2 = True, load = False)