from .peaks import *
from ..models2.helper import get_nf
from ..finder import fatigue_data
import numpy as np
import pandas as pd

all_const_data = ['temp', 'strain', 'rate']
const_data = ['temp', 'strain', 'rate']

def drop_time_feats(Xv, feats):
    Xv_new = np.array([x.drop(feats, axis = 1) for x in Xv], dtype=object) 
    return Xv_new

def drop_const_feats(Xc, feats):
        return Xc.drop(feats, axis = 1)

def vectorise_data(data = fatigue_data.data, tfeats = [], cfeats = []):
    X_vary = []
    X_const = []
    y = []
    for test in data:
        tempX = features(test).reset_index(drop = True)
        X_vary.append(tempX.drop(all_const_data, axis = 1))
        X_const.append(tempX[const_data].iloc[0]) 
        y.append(get_nf(test))
    Xv = np.array(X_vary, dtype = object)
    Xc = pd.DataFrame(X_const, columns=all_const_data).reset_index(drop = True)
    y = np.array(y).reshape(-1, 1)
    
    if tfeats:
        Xv = drop_time_feats(Xv, tfeats)
    if cfeats:
        Xc = drop_const_feats(Xc, cfeats)
    return Xv, Xc, y           

def single_input_data(data = fatigue_data.data):
    X = []
    y = []
    for test in data:
        tempX = features(test).reset_index(drop = True)
        X.append(tempX.drop(all_const_data, axis = 1))
        y.append(get_nf(test))
    return np.array(X, dtype = object), np.array(y).reshape(-1, 1)

def slatten(data = fatigue_data.data, tfeats = [], cfeats = [], cycles = 10):
    Xv, Xc, y = vectorise_data(data, tfeats=tfeats, cfeats=cfeats)
    
    Xv_proc = []
    
    for x in Xv:
        Xv_proc.append(x.iloc[:cycles].to_numpy().flatten())
    
    Xv = np.array(Xv_proc)
    
    return np.concatenate((Xv, Xc), axis = 1), y

def natural(data = fatigue_data.data, tfeats = [], cfeats = [], cycles = 10):
    X_vary = []
    X_const = []
    y = []
    for test in data:
        tempX = features_nat(test).iloc[:cycles].reset_index(drop = True)
        X_vary.append(tempX.drop(all_const_data, axis = 1))
        X_const.append(tempX[const_data].iloc[0]) 
        y.append(get_nf(test))
    Xv = X_vary
    Xc = pd.DataFrame(X_const, columns=all_const_data).reset_index(drop = True)
    y = np.array(y).reshape(-1, 1)
    
    if tfeats:
        Xv = drop_time_feats(Xv, tfeats)
    if cfeats:
        Xc = drop_const_feats(Xc, cfeats)
    return Xv, Xc, y

