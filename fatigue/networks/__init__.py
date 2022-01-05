from .peaks import *
from ..models2.helper import get_nf
from ..finder import fatigue_data
import numpy as np

const_data = ['temp', 'strain', 'rate']

def vectorise_data(data = fatigue_data.data, cycles = False):
    X_vary = []
    X_const = []
    y = []
    for test in data:
        tempX = features(test).reset_index(drop = True)
        X_vary.append(tempX.drop(const_data, axis = 1))
        X_const.append(tempX[const_data].iloc[0]) 
        y.append(get_nf(test))
    return np.array(X_vary, dtype='object'), np.array(X_const), y
    
def ragged_numpy_arr(rlist):
    return np.array([np.array(el) for el in rlist], dtype= 'object')