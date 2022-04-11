import numpy as np
from scipy.optimize import curve_fit
from .helper import get_nf, TEMPS

from ..strain import total_elastic_strain_percent
import sys

sys.path.append('/../../')

from temp.get_folds import Data, test_idx, train_idx

def cmanson_eqn(x, A, a):
	return A*(x)**a

def cmanson_eqn_log(x, A, a):
	return A*x + a
        
def cmanson_pred(y, A, a):
	return (y/A)**(1/a)

def cmanson_construct(fatigue_data):
    xx = []; tt= []
    funcs = []
    log_coeffs = []
    train = Data.iloc[train_idx['best']]
    test = Data.iloc[test_idx['best']]
    
    for temp in TEMPS:
        x = np.array([])
        y = np.array([])
        tmp = train[train.Temps == temp]
        for i in range(len(tmp)):
            x = np.append(x, tmp.iloc[i].Cycles)
            test_, = fatigue_data.get_test_from_sample(tmp.iloc[i].Samples)
            y = np.append(y, total_elastic_strain_percent(test_))
        print(f'Data for {temp}C acquired')
        xx.append([x, y])
        
        x = np.array([])
        y = np.array([])
        tmp = test[test.Temps == temp]
        for i in range(len(tmp)):
            x = np.append(x, tmp.iloc[i].Cycles)
            test_, = fatigue_data.get_test_from_sample(tmp.iloc[i].Samples)
            y = np.append(y, total_elastic_strain_percent(test_))
        print(f'Data for {temp}C acquired')
        tt.append([x, y])
    
    for i, (x, y) in enumerate(xx):
        popt, _ = curve_fit(cmanson_eqn_log, np.log10(x), np.log10(y), p0 = [-0.95, -2.4])
        a = popt[0]; A = 10**popt[1]
        log_coeffs.append([A, a])
        funcs.append(lambda x, A = A, a = a: cmanson_eqn(x, A, a))
        
    return xx, tt, funcs, log_coeffs
        