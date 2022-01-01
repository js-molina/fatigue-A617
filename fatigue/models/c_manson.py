import numpy as np
from scipy.optimize import curve_fit
from .helper import get_nf, TEMPS

from ..strain import total_elastic_strain_percent

def cmanson_eqn(x, A, a):
	return A*(x)**a

def cmanson_eqn_log(x, A, a):
	return A*x + a
        
def cmanson_pred(y, A, a):
	return (y/A)**(1/a)

def cmanson_construct(fatigue_data):
    xx = []
    funcs = []
    log_coeffs = []
    for temp in TEMPS:
        x = np.array([])
        y = np.array([])
        for test in fatigue_data.get_data(temp):
            x = np.append(x, get_nf(test))
            y = np.append(y, total_elastic_strain_percent(test))
        print(f'Data for {temp}C acquired')
        xx.append([x, y])
        
    for i, (x, y) in enumerate(xx):
        popt, _ = curve_fit(cmanson_eqn_log, np.log10(x), np.log10(y), p0 = [-0.95, -2.4])
        a = popt[0]; A = 10**popt[1]
        log_coeffs.append([A, a])
        funcs.append(lambda x, A = A, a = a: cmanson_eqn(x, A, a))

    return xx, funcs, log_coeffs
        