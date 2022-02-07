import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)

sys.path.append('..')

import sklearn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from fatigue.networks import slatten
from fatigue.graph.models2 import graph_nn_pred_all
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams['text.usetex'] = False

tcols = ['max_s_m', 'min_s_m', 's_ratio_m', 'elastic_m', 'plastic_m',
       'max_s_d_m', 'min_s_d_m', 's_ratio_d_m', 'elastic_d_m', 'plastic_d_m']

ccols = ['temp', 'strain', 'rate']

tfts = ['plastic_d_m', 's_ratio_m', 's_ratio_d_m']
cfts = ['rate']

tfts, cfts = [], []

tcols = list(set(tcols) - set(tfts))
ccols = list(set(ccols) - set(cfts))

def report_coeff(names, coef, intercept):
    
    gcoef = []
    for i in range(tfeats):
        t = np.arange(i, ncycles*tfeats+i, tfeats)
        gcoef.append(coef[t].mean())
        
    for i in coef[-len(ccols):]:
         gcoef.append(i)
         
    gcoef = np.array(gcoef)
    
    r = pd.DataFrame({'coef': gcoef, 'positive': gcoef >= 0}, index = tcols + ccols)
    r = r.sort_values(by = ['coef'])
    r['coef'].plot(kind = 'barh', color = r['positive'].map({True: 'b', False: 'r'}))

    plt.show()
    return r

ncycles = 60
tfeats = len(tcols)

columns = []

for i in range(ncycles):
    columns += [s + '[%d]'%(i+1) for s in tcols]

columns += ccols

drop_strain = True
drop_temp = True

x, y = slatten(cycles = ncycles, tfeats=tfts, cfeats=cfts)

x = pd.DataFrame(x, columns=columns)

if drop_strain:
    ccols = ['temp', 'rate']
    x = x.drop('strain', axis = 1)
    
if drop_temp:
    ccols = ['rate']
    x = x.drop('temp', axis = 1)

y = np.log1p(y) 

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=30)

xScaler = StandardScaler()
xScaler.fit(x_train)

yScaler = StandardScaler()
yScaler.fit(y_train)

X_train, X_test = map(xScaler.transform, [x_train, x_test])
Y_train, Y_test = map(yScaler.transform, [y_train, y_test])

model = LinearRegression()

model.fit(X_train, Y_train)

pred0 = model.predict(X_train)
Y_obs0, Y_pred0 = map(yScaler.inverse_transform, [Y_train, pred0])
y_obs0, y_pred0 = map(np.expm1, [Y_obs0, Y_pred0])

pred1 = model.predict(X_test)
Y_obs1, Y_pred1 = map(yScaler.inverse_transform, [Y_test, pred1])
y_obs1, y_pred1 = map(np.expm1, [Y_obs1, Y_pred1])

score0 = sklearn.metrics.mean_absolute_percentage_error(y_obs0, y_pred0)
score1 = sklearn.metrics.mean_absolute_percentage_error(y_obs1, y_pred1)

print(f'Final MAPE: {score0:.3f}/{score1:.3f}')

err = abs(y_obs1-y_pred1)/y_obs1*100

names = x.columns

r = report_coeff(names, model.coef_[0,:], model.intercept_)

# nr_data = {'y_obs_test': np.concatenate((y_obs1, y_obs0)), 'y_pred_test': np.concatenate((y_pred1,y_pred0))}

# nr_data = {'y_obs_test': y_obs1,'y_pred_test': y_pred1}

# graph_nn_pred_all(nr_data, log = True, v2 = True, load = False)

#%%

model = ElasticNet(random_state=0, alpha = 0.2)

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

# r_data = {'y_obs_test': np.concatenate((y_obs1, y_obs0)),'y_pred_test': np.concatenate((y_pred1,y_pred0))}

# graph_nn_pred_all(r_data, log = True, v2 = True, load = False)

r = report_coeff(names, model.coef_, model.intercept_)


