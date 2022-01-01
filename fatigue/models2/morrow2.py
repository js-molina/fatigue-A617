import numpy as np
from scipy.optimize import curve_fit
from .helper import get_nf, TEMPS

from ..strain import total_plastic_energy

def morrow_eqn2(x, A, a):
	return A*(x)**a

def morrow_eqn_log2(x, A, a):
	return A*x + a
        
def morrow_construct2(fatigue_data):
    xx = []
    funcs = []
    log_coeffs = []
    for temp in TEMPS:
        x = np.array([])
        y = np.array([])
        for test in fatigue_data.get_data(temp):
            x = np.append(x, total_plastic_energy(test))
            y = np.append(y, get_nf(test))
        print(f'Data for {temp}C acquired')
        xx.append([x, y])
        
    for i, (x, y) in enumerate(xx):
        popt, _ = curve_fit(morrow_eqn_log2, np.log10(x), np.log10(y))
        a = popt[0]; A = 10**popt[1]
        log_coeffs.append([A, a])
        funcs.append(lambda x, A = A, a = a: morrow_eqn2(x, A, a))

    return xx, funcs, log_coeffs
        