import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from functools import partial
import matplotlib, os, sys

sys.path.append('..')

from scipy.optimize import root, fsolve
from get_folds import Data
from fatigue.finder import fatigue_data
from fatigue.finder.cycle_path import peak_path_from_sample
from fatigue.strain import cycle_plastic_strain_percent, cycle_elastic_strain_percent
from fatigue.graph.models2 import graph_nn_pred_all, graph_nn_1_fold, graph_nn_2_fold, graph_nn_hist, get_meap

from sklearn.metrics import r2_score

matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{siunitx,amsmath, amssymb, amsfonts, amsthm}'

TEMPS = [850, 950]

half_live_labels = ['46%', '47%', '48%', '49%', '50%', '51%', '52%', '53%', '54%', '55%', '56%', '57%']

def lin_to_log(m, a):
    return lambda x: np.exp(a)*(x)**m

def func(x, a, b):
    return a*np.log(x[0]) + np.log(b*x[1]/x[2])

#%%

fig = plt.figure(figsize=(7,3))
ax = [fig.add_axes([0, 0, 3/7, 1]), fig.add_axes([3.8/7, 0, 3/7, 1])]
mcs = []; mcsd = []

ylims = [(1, 1e2), (1, 1e4)]

E = [153e3, 144e3]

LF = []

for i, t in enumerate(TEMPS):
    DATA = Data[Data.Temps == t]

    cycle_times = []
    cycles_to_failure = []
    elastic_strain = []
    plastic_strain = []
    total_strain = []

    for sample in DATA.Samples:
        tmp = pd.read_csv(peak_path_from_sample(sample))
        tx = tmp.iloc[tmp.Cycle.iloc[-1]//2]
        max_s, min_s =  map(float, tx[['Max Stress Mpa', 'Min Stress Mpa']])
        s_ratio = abs(max_s/min_s)
        total = float(DATA[DATA.Samples == sample].Strains/100)
        elastic = (max_s-min_s)/E[i]
        plastic = total - elastic
        cycles = int(DATA[DATA.Samples == sample].Cycles)
    
        plastic_strain.append(plastic)
        total_strain.append(total)
        cycles_to_failure.append(cycles)
        cycle_times.append(total/0.0001)

    ax[i].plot(np.log10(cycles_to_failure), np.log10(cycle_times), 'bo', markerfacecolor = 'None', markeredgewidth = 1)

    line_fit = np.polyfit(np.log10(np.array(cycles_to_failure)), np.log10(cycle_times), 1)
    
    LF.append(line_fit[0])
    
    xlim = np.array([min(cycles_to_failure), max(cycles_to_failure)])
    x = np.linspace(*np.log10(xlim))
    
    ax[i].plot(x, np.poly1d(line_fit)(x), 'b--', lw = 1)
    
    # ax[i].set_xscale('log')
    # ax[i].set_yscale('log')
    
    # ax[i].set_xlim(1e2, 2e4)
    # ax[i].set_ylim(ylims[i])
    
    ax[i].set_ylabel(r'Cycle Time')
    ax[i].set_xlabel(r'Cycles to Failure, $N_f$')

    ax[i].grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
plt.show()

#%%

obs = []
pred = []

for i, t in enumerate(TEMPS):
    DATA = Data[Data.Temps == t]
    half_live_cycles = []

    elastic_strain = []
    plastic_strain = []
    total_strain = []
    cycles_to_failure = []
    strain_rate = []
    stress_range = []

    for sample in DATA.Samples:
        tmp = pd.read_csv(peak_path_from_sample(sample))
        tx = tmp.iloc[tmp.Cycle.iloc[-1]//2]
        max_s, min_s =  map(float, tx[['Max Stress Mpa', 'Min Stress Mpa']])
        s_ratio = abs(max_s/min_s)
        total = float(DATA[DATA.Samples == sample].Strains/100)
        elastic = (max_s-min_s)/E[i]
        plastic = total - elastic
        cycles = int(DATA[DATA.Samples == sample].Cycles)
    
        strain_rate.append(float(DATA[DATA.Samples == sample].Rates))
        stress_range.append(max_s-min_s)        
        elastic_strain.append(elastic)
        plastic_strain.append(plastic)
        total_strain.append(total)
        cycles_to_failure.append(cycles)
    
    pl = np.array(plastic_strain)
    tl = np.array(total_strain)
    sr = np.array(strain_rate)
    ss = np.array(stress_range)
    ra = (tl/sr)**LF[i]
    params = np.array([pl, ra, ss])
    
    popt, pcov = curve_fit(func, params, np.log(cycles_to_failure))
    
    pred += (popt[1]*(ra*pl**popt[0]/ss)).tolist()
    obs += cycles_to_failure

#%%

obs = np.array(obs)
pred = np.array(pred)

r_data = {'y_obs_train': obs[:33], 'y_pred_train': pred[:33], 'y_obs_test': obs[33:], 'y_pred_test': pred[33:]}

graph_nn_1_fold(r_data, log = True, load = False, which = 'both')
print(get_meap(r_data, load = False, v2 = False))
