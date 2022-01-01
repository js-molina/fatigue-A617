from ..graph.helper import *
from scipy.signal import savgol_filter
import numpy as np

RATIO = 12

def filter_by_savgol(data, ratio = RATIO, deg = 1):
    window = len(data)//ratio
    if window%2 == 1:
        return savgol_filter(data, window, deg)
    else:
        return savgol_filter(data, window+1, deg)

def scuffed_zeros(c1):

    strain = c1['Strain']
    stress = c1['Stress Mpa']
    
    eps = 1e-5
    
    p1 = c1.loc[(strain > eps) & (stress > 0)].sort_values(by = 'Stress Mpa').iloc[0]
    
    p2 = c1.loc[(strain > eps) & (stress < 0)]
    
    if len(p2) > 0:
        p2 = p2.sort_values(by = 'Stress Mpa').iloc[-1]
    else:
        p2 = p1
    
    p3 = c1.loc[(strain < -eps) & (stress > 0)]
    
    if len(p3) > 0:
        p3 = p3.sort_values(by = 'Stress Mpa').iloc[0]
    else:
        p3 = c1.iloc[0]
        
    p4 = c1.loc[(strain < -eps) & (stress < 0)]
    
    if len(p4) > 0:
        p4 = p4.sort_values(by = 'Stress Mpa').iloc[-1]
    else:
        p4 = c1.iloc[0]

    xx = np.array([p3.Strain+p4.Strain, p1.Strain+p2.Strain])/2
    
    return xx

def get_plastic_zeros(cycle):
    
    idx = np.argwhere(np.diff(np.sign(cycle['Stress Mpa']))).flatten()

    if len(idx) != 2:
        idx = np.argwhere(np.diff(np.sign(filter_by_savgol(cycle['Stress Mpa'], RATIO, 2)))).flatten()
        if len(idx) < 2:
            return scuffed_zeros(cycle)

    x0 = cycle.Strain.iloc[idx].values; y0 = cycle['Stress Mpa'].iloc[idx].values
    x1 = cycle.Strain.iloc[idx+1].values; y1 = cycle['Stress Mpa'].iloc[idx+1].values

    p0 = np.poly1d(np.polyfit([y0[0], y1[0]], [x0[0], x1[0]], 1))
    p1 = np.poly1d(np.polyfit([y0[1], y1[1]], [x0[1], x1[1]], 1))

    xx = [p0(0), p1(0)]

    return scuffed_zeros(cycle)

def cycle_plastic_strain_percent(c):
    zeros = get_plastic_zeros(c)
    return np.diff(zeros)[0]*100

def total_plastic_strain_percent(test):
    
    cycles = get_cycles_from_test(test)
    strain_vals = []
    
    for cycle in cycles:
        strain_vals.append(cycle_plastic_strain_percent(cycle))
        
    return np.mean(strain_vals[-3:])

def cycle_elastic_strain_percent(c, t):
    
    if t == 850:
        E = 153e3
    else:
        E = 144e3 

    stress_max, stress_min = max(c['Stress Mpa']), min(c['Stress Mpa'])
    stress_range = stress_max - stress_min
    
    return stress_range/E*100

def total_elastic_strain_percent(test):
    
    cycles = get_cycles_from_test(test)
    strain_vals = []
    
    if test.Temp == 850:
        E = 153e3
    else:
        E = 144e3 
    
    for cycle in cycles:
        stress_max, stress_min = max(cycle['Stress Mpa']), min(cycle['Stress Mpa'])
        stress_range = stress_max - stress_min
        strain_vals.append(stress_range/E)
        
    return np.mean(strain_vals[-3:])*100

def get_plastic_energy(c1):
    
    dx = np.diff(c1.Strain)

    ytmp = c1['Stress Mpa'].tolist()

    j = 0
    dy = []
    while j < len(ytmp)-1:
        dy.append(np.mean(ytmp[j:j+1]))
        j += 1
        
    return(sum(dx*dy))

def total_plastic_energy(test):
        
    cycles = get_cycles_from_test(test)
    energy_vals = []
        
    for c1 in cycles:
        energy_vals.append(get_plastic_energy(c1))
        
    return np.mean(energy_vals[-3:])

def scuffed_plastic_energy_pts(s_max, s_min, pl, e):
    
    x1 = pl/2; y1 = 0
    x2 = e/2; y2 = s_max
    y3 = s_max
    x4 = -x1; y4 = 0
    x5 = -x2; y5 = s_min
    y6 = s_min
    
    x3 = np.poly1d(np.polyfit([y4, y5], [x4, x5], 1))(y3)
    x6 = np.poly1d(np.polyfit([y1, y2], [x1, x2], 1))(y6)
    
    return np.array([x1, x2, x3, x4, x5, x6]), np.array([y1, y2, y3, y4, y5, y6])

