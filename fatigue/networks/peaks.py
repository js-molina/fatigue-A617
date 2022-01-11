from ..graph import get_peak_data_from_test
import numpy as np
# from ..models2 import get_nf

def stress(test):
    x = get_peak_data_from_test(test).copy()
    x = x[['Cycle', 'Max Stress Mpa', 'Min Stress Mpa', 'Stress Ratio']]
    x.columns = ['cycle', 'max_s', 'min_s', 's_ratio']
    return x

def app_elastic_e(x, E):
    x = x.assign(elastic = (x.max_s-x.min_s)/E*100)
    return x

def app_plastic_e(x, e):
    x = x.assign(plastic = e - x.elastic)
    return x

def app_diff(x):
    x = x.assign(max_diff = np.insert(np.diff(x.max_s), values = 0, obj = 0))
    x = x.assign(min_diff = np.insert(np.diff(x.min_s), values = 0, obj = 0))
    x = x.assign(s_ratio_diff = np.insert(np.diff(x.s_ratio), values = 0, obj = 0))
    return x

def app_int(x):
    x = x.assign(max_int = x.max_s.expanding(1).sum())
    x = x.assign(min_int = x.min_s.expanding(1).sum())
    x = x.assign(s_ratio_int = x.s_ratio.expanding(1).sum())
    return x
           
def app_bavg(x):
    
    l = len(x)
    
    vals = [100, 50, 10]
    
    strs = ['b%d'%n for n in vals]
    
    for n, s in zip(vals, strs):
        tmp = x.rolling(n, min_periods=1).mean()
        kwargs = {'max_s_' + s : tmp.max_s, 'min_s_' + s : tmp.min_s,
                  'max_diff_' + s : tmp.max_diff, 'min_diff_' + s : tmp.min_diff}
        x = x.assign(**kwargs)
    
    vals = [30, 20, 10]
    strs = ['b%dp'%n for n in vals]
    
    for n, s in zip(vals, strs):
        tmp = x.rolling(l*n//100, min_periods=1).mean()
        kwargs = {'max_s_' + s : tmp.max_s, 'min_s_' + s : tmp.min_s}
        x = x.assign(**kwargs)

    return x

def app_favg(x):
    
    l = len(x)
    
    vals = [100, 50, 10]
    
    strs = ['f%d'%n for n in vals]
    
    for n, s in zip(vals, strs):
        tmp = x[::-1].rolling(n, min_periods=1).mean()[::-1]
        kwargs = {'max_s_' + s : tmp.max_s, 'min_s_' + s : tmp.min_s}
        x = x.assign(**kwargs)
    
    vals = [30, 20, 10]
    strs = ['f%dp'%n for n in vals]
    
    for n, s in zip(vals, strs):
        tmp = x[::-1].rolling(l*n//100, min_periods=1).mean()[::-1]
        kwargs = {'max_s_' + s : tmp.max_s, 'min_s_' + s : tmp.min_s}
        x = x.assign(**kwargs)
    
    return x  

def app_const(x, test):
    x['temp'] = test.Temp
    x['strain'] = test.Strain
    x['rate'] = test.Rate
    
def features(test, cycles = False):
    
    x = stress(test)
    
    if test.Temp == 850:
        E = 153e3
    else:
        E = 144e3 
    
    x = app_diff(x)
    
    x = app_elastic_e(x, E)
    x = app_plastic_e(x, test.Strain)
    
    x = app_bavg(x)
    x = app_favg(x)
    
    app_const(x, test)
    
    if not cycles:
        x = x.drop('cycle', axis = 1)
    
    s = np.random.randint(0, 2)

    return x.iloc[10:120+s]