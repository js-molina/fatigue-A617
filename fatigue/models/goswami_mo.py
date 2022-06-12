import numpy as np
from scipy.optimize import curve_fit
from fatigue.finder.cycle_path import peak_path_from_sample
from .helper import get_nf, TEMPS

from ..strain import total_plastic_strain_percent
import pandas as pd
import sys

sys.path.append('/../../')

from temp.get_folds import Data, test_idx, train_idx

def goswami_eqn(x, a, b):
	return a*np.log(x[0]) + np.log(b*x[1]/x[2])
        
def goswami_pred(x, m, n, K):
	return K*x[0]**(n)*(x[1]/x[2])**m/x[3]

def goswami_construct(fatigue_data):
    E = [153e3, 144e3]    
    xx = []; tt= []
    funcs = []
    log_coeffs = []
    train = Data.iloc[train_idx['best']]
    test = Data.iloc[test_idx['best']]
    
    for j, temp in enumerate(TEMPS):
        x = np.array([])
        ep, et, r, ss = [], [], [], []
        tmp = train[train.Temps == temp]
        for i in range(len(tmp)):
            x = np.append(x, tmp.iloc[i].Cycles)
            tm = pd.read_csv(peak_path_from_sample(tmp.iloc[i].Samples))
            tx = tm.iloc[tm.Cycle.iloc[-1]//2]
            max_s, min_s =  map(float, tx[['Max Stress Mpa', 'Min Stress Mpa']])
            total = tmp.iloc[i].Strains/100
            rate = tmp.iloc[i].Rates
            elastic = (max_s-min_s)/E[j]
            plastic = total - elastic
            ss.append(max_s-min_s)
            et.append(total)
            ep.append(plastic)
            r.append(rate)
        print(f'Data for {temp}C acquired')
        xx.append(list(map(np.array, [x, ep, et, r, ss])))
        
        x = np.array([])
        ep, et, r, ss = [], [], [], []
        tmp = test[test.Temps == temp]
        for i in range(len(tmp)):
            x = np.append(x, tmp.iloc[i].Cycles)
            tm = pd.read_csv(peak_path_from_sample(tmp.iloc[i].Samples))
            tx = tm.iloc[tm.Cycle.iloc[-1]//2]
            max_s, min_s =  map(float, tx[['Max Stress Mpa', 'Min Stress Mpa']])
            total = tmp.iloc[i].Strains/100
            rate = tmp.iloc[i].Rates
            elastic = (max_s-min_s)/E[j]
            plastic = total - elastic
            ss.append(max_s-min_s)
            et.append(total)
            ep.append(plastic)
            r.append(rate)
        print(f'Data for {temp}C acquired')
        tt.append(list(map(np.array, [x, ep, et, r, ss])))
    
    for i, (x, ep, et, r, ss) in enumerate(xx):
        
        m, _  = np.polyfit(np.log10(x), np.log10(et/r), 1)
        
        ra = (et/r)**m
        
        params = np.array([ep, ra, ss])
        
        popt, _ = curve_fit(goswami_eqn, params, np.log(x))
        n = popt[0]; K = popt[1]
        log_coeffs.append([m, n, K])
        funcs.append(lambda x, n = n, K = K: goswami_eqn(x, n, K))
        
    return xx, tt, funcs, log_coeffs