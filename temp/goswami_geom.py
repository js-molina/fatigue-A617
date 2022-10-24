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
from fatigue.graph.models2 import graph_nn_pred_all, graph_nn_1_fold, graph_nn_2_fold, graph_nn_12_dev, graph_nn_hist, get_meap, get_chi

from sklearn.metrics import r2_score
import tdt


matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{siunitx,amsmath, amssymb, amsfonts, amsthm}'

TEMPS = [850, 950, 'a']

half_live_labels = ['46%', '47%', '48%', '49%', '50%', '51%', '52%', '53%', '54%', '55%', '56%', '57%']

def lin_to_log(m, a):
    return lambda x: np.exp(a)*(x)**m

def func(x, a, b):
    return a*np.log(x[0]) + np.log(b*x[1]/x[2])

#%%

fig, tax = plt.subplots(3, 1, figsize=(3,9))
# ax = [fig.add_axes([0, 0, 3/7, 1]), fig.add_axes([3.8/7, 0, 3/7, 1])]
ax = [tax[i] for i in range(3)]
mcs = []; mcsd = []

ylims = [(1, 1e2), (1, 1e4)]

E = [153e3, 144e3, 148.5e3]

LF = []


for i, t in enumerate(TEMPS):
    DATA = Data[(Data.Temps == t)] if t != 'a' else Data

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
        cycle_times.append(total/float(DATA[DATA.Samples == sample].Rates))

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
    DATA = Data[(Data.Temps == t)] if t != 'a' else Data
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
    
    pl = np.array(plastic_strain)
    tl = np.array(total_strain)
    sr = np.array(strain_rate)
    ss = np.array(stress_range)
    ra = (tl/sr)**LF[i]
    
    pred.append((popt[1]*(ra*pl**popt[0]/ss)))
    obs.append(np.array(cycles_to_failure))

#%%

obs = np.array(obs, dtype= 'object')
pred = np.array(pred, dtype= 'object')

ax = plt.gca()

ax.set_aspect('equal')

ax.set_ylim(100, 20000)
ax.set_xlim(100, 20000)

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlabel('Predicted $N_f$')
ax.set_ylabel('Measured $N_f$')

ax.plot([100, 20000], [100, 20000], lw = 2, color = 'k')
ax.fill_between([100, 20000], 100, [100, 20000], color = 'k', alpha = 0.1)

ax.plot([100, 1e5], [200, 2e5], lw = 1, ls = '--', color = 'gray')
ax.plot([200, 2e5], [100, 1e5], lw = 1, ls = '--', color = 'gray')

ax.plot(pred[0], obs[0], 'bx', markersize = 7, ls = 'None', \
        markerfacecolor = 'None', markeredgewidth = 2, label = r'\SI{850}{\celsius} -- $%.1f$'%((abs(obs[0]-pred[0])/obs[0]).mean()*100) + '\%')
ax.plot(pred[1], obs[1], 'ro', markersize = 7, ls = 'None', markerfacecolor = 'None', \
        markeredgewidth = 2, label = r'\SI{950}{\celsius} -- $%.2f$'%((abs(obs[1]-pred[1])/obs[1]).mean()*100) + '\%')

ax.plot(pred[2], obs[2], 's', markersize = 7, ls = 'None', markerfacecolor = 'None', markeredgecolor = '#ff9900', \
    markeredgewidth = 2, label = r'Hybrid -- $%.2f$'%((abs(obs[2]-pred[2])/obs[2]).mean()*100) + '\%')

    
ax.tick_params(axis = 'both', direction='in', which = 'both')
ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
ax.set_title(r'\textbf{Goswami}')

legend = ax.legend(edgecolor = 'k')
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((1, 1, 1, 1))

props = dict(boxstyle='round', facecolor= (0.9, 0.9, 0.9), lw = 1)
ax.text(0.95, 0.05, r'\textit{Non-Conservative Region}', transform=ax.transAxes, va='bottom', ha = 'right', bbox=props)

plt.savefig(os.path.join(r'D:\INDEX\Notes\Semester_15\MMAN4952\Thesis B\figs', 'goswami.pdf'), bbox_inches = 'tight')
plt.show()  

