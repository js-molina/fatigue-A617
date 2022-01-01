import numpy as np
from functools import partial
from scipy.optimize import curve_fit

def func(t, x, a, b, c):
    return a*np.exp(-b*x) + c + t

x = np.linspace(0,4,50)

pfunc = partial(func, 0.0)
y = pfunc(x, 2.5, 1.3, 0.5)
popt, pcov = curve_fit(pfunc, x, y)
print(popt, y[0:2])

pfunc = partial(func, a = 1.5)
y = pfunc(x, t = 2.5, b = 1.3, c = 0.5)
popt, pcov = curve_fit(pfunc, x, y)
print(popt, y[0:2])