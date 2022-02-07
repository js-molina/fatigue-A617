from ..graph import get_peak_data_from_test
import numpy as np
# from ..models2 import get_nf

def stress(test):
    x = get_peak_data_from_test(test).copy()
    x = x[['Max Stress Mpa', 'Min Stress Mpa', 'Stress Ratio']]
    x.columns = ['max_s', 'min_s', 's_ratio']
    return x.reset_index(drop = True)

def app_elastic_e(x, E):
    x = x.assign(elastic = (x.max_s-x.min_s)/E*100)
    return x

def app_plastic_e(x, e):
    x = x.assign(plastic = e - x.elastic)
    return x

def app_diff(x):
    tmp = x.diff()
    tmp.loc[0] = 0
    kwarg = {}
    for col in tmp.columns:
        kwarg[col + '_d'] = tmp[col]
    x = x.assign(**kwarg)    
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

def app_avg(x):
    tmp = x.rolling(10, min_periods=1, center=True).mean()
    kwarg = {}
    for col in tmp.columns:
        kwarg[col + '_m'] = tmp[col]
    x = x.assign(**kwarg)    
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
    
    x = app_elastic_e(x, E)
    x = app_plastic_e(x, test.Strain)
    
    x = app_diff(x)
    
    cols = x.columns    

    x = app_avg(x)
    
    x = x.drop(cols, axis = 1)
    
    app_const(x, test)
    
    return x.iloc[3:]

def features_rel(test, cycles = False):
    
    x = stress(test)
    
    if test.Temp == 850:
        E = 153e3
    else:
        E = 144e3 
    
    x = app_elastic_e(x, E)
    x = app_plastic_e(x, test.Strain)
    
    cols = x.columns  
    
    x = app_diff(x)
    
    x = x.drop(cols, axis = 1)

    x = app_avg(x)
    
    app_const(x, test)
    
    return x.iloc[3:]

def features_nat(test, cycles = False):
    
    x = stress(test)
    
    if test.Temp == 850:
        E = 153e3
    else:
        E = 144e3 
    
    x = app_elastic_e(x, E)
    x = app_plastic_e(x, test.Strain)
    
    app_const(x, test)
    
    return x.iloc[3:]