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

fold = 'best'
train, dev, test = tdt.train_idx[fold], tdt.dev_idx[fold], tdt.test_idx[fold]

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

# D_train = Data.loc[train]
# D_dev = Data.loc[dev]
# D_test = Data.loc[test]

train_dev = np.concatenate((train, dev))

D_train = Data.loc[train_dev]
D_test = Data.loc[test]

for i, t in enumerate(TEMPS):
    DATA = D_train[(Data.Temps == t)] if t != 'a' else D_train

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

obs_test = []
pred_test = []

for i, t in enumerate(TEMPS):
    DATA = Data[(Data.Temps == t)] if t != 'a' else Data
    half_live_cycles = []

    elastic_strain = []
    plastic_strain = []
    total_strain = []
    cycles_to_failure = []
    strain_rate = []
    stress_range = []
    
    elastic_strain_test = []
    plastic_strain_test = []
    total_strain_test = []
    cycles_to_failure_test = []
    strain_rate_test = []
    stress_range_test = []
    
    for sample in DATA.Samples:
        tmp = pd.read_csv(peak_path_from_sample(sample))
        tx = tmp.iloc[tmp.Cycle.iloc[-1]//2]
        max_s, min_s =  map(float, tx[['Max Stress Mpa', 'Min Stress Mpa']])
        s_ratio = abs(max_s/min_s)
        total = float(DATA[DATA.Samples == sample].Strains/100)
        elastic = (max_s-min_s)/E[i]
        plastic = total - elastic
        cycles = int(DATA[DATA.Samples == sample].Cycles)
        
        if sample in D_train.Samples.tolist():
            strain_rate.append(float(DATA[DATA.Samples == sample].Rates))
            stress_range.append(max_s-min_s)        
            elastic_strain.append(elastic)
            plastic_strain.append(plastic)
            total_strain.append(total)
            cycles_to_failure.append(cycles)
        else:
            strain_rate_test.append(float(DATA[DATA.Samples == sample].Rates))
            stress_range_test.append(max_s-min_s)        
            elastic_strain_test.append(elastic)
            plastic_strain_test.append(plastic)
            total_strain_test.append(total)
            cycles_to_failure_test.append(cycles)
    
    pl = np.array(plastic_strain)
    tl = np.array(total_strain)
    sr = np.array(strain_rate)
    ss = np.array(stress_range)
    ra = (tl/sr)**LF[i]
    params = np.array([pl, ra, ss])
    
    popt, pcov = curve_fit(func, params, np.log(cycles_to_failure))
    
    pl_test = np.array(plastic_strain_test)
    tl_test = np.array(total_strain_test)
    sr_test = np.array(strain_rate_test)
    ss_test = np.array(stress_range_test)
    ra_test = (tl_test/sr_test)**LF[i]
    
    pred.append((popt[1]*(ra*pl**popt[0]/ss)))
    obs.append(np.array(cycles_to_failure))

    pred_test.append((popt[1]*(ra_test*pl_test**popt[0]/ss_test)))
    obs_test.append(np.array(cycles_to_failure_test))
    
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

ax.plot(pred[0], obs[0], 'bo', markersize = 7, ls = 'None', \
        markerfacecolor = 'None', markeredgewidth = 2, label = r'\SI{850}{\celsius} -- Train, (%.1f'%((abs(obs[0]-pred[0])/obs[0]).mean()*100) + '\%)')
ax.plot(pred_test[0], obs_test[0], 'bx', markersize = 7, ls = 'None', \
        markerfacecolor = 'None', markeredgewidth = 2, label = r'\SI{850}{\celsius} -- Test, (%.1f'%((abs(obs_test[0]-pred_test[0])/obs_test[0]).mean()*100) + '\%)')
    
ax.plot(pred[1], obs[1], 'ro', markersize = 7, ls = 'None', markerfacecolor = 'None', \
        markeredgewidth = 2, label = r'\SI{950}{\celsius} -- Train, (%.1f'%((abs(obs[1]-pred[1])/obs[1]).mean()*100) + '\%)')
ax.plot(pred_test[1], obs_test[1], 'rx', markersize = 7, ls = 'None', markerfacecolor = 'None', \
        markeredgewidth = 2, label = r'\SI{950}{\celsius} -- Test, (%.1f'%((abs(obs_test[1]-pred_test[1])/obs_test[1]).mean()*100) + '\%)')
    
ax.tick_params(axis = 'both', direction='in', which = 'both')
ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
ax.set_title(r'\textbf{Goswami}')

legend = ax.legend(edgecolor = 'k', loc = 2)
# legend.get_frame().set_alpha(0)
# legend.get_frame().set_facecolor((1, 1, 1, 1))

props = dict(boxstyle='round', facecolor= (0.9, 0.9, 0.9), lw = 1)
ax.text(0.95, 0.05, r'\textit{Non-Conservative Region}', transform=ax.transAxes, va='bottom', ha = 'right', bbox=props)

path = r'D:\INDEX\Notes\Semester_15\MMAN4952\Thesis B\figs'
path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'

# plt.savefig(os.path.join(path, 'goswami_split.pdf'), bbox_inches = 'tight')

np.savez('../mdata/gos850tt', y_pred_train = pred[0], y_obs_train = obs[0],
                            y_pred_test = pred_test[0], y_obs_test = obs_test[0])
np.savez('../mdata/gos950tt', y_pred_train = pred[1], y_obs_train = obs[1],
                            y_pred_test = pred_test[1], y_obs_test = obs_test[1])

plt.show()  

# %%
final_table = pd.read_csv(r'../mdata/final.csv')

# final_table = Data.copy()

all_obs = list(np.concatenate(np.concatenate((obs[:2], obs_test[:2]))))
all_pred = np.concatenate(np.concatenate((pred[:2], pred_test[:2])))

ord_pred = []

for c in Data.Cycles:
    i = all_obs.index(c)
    print(all_obs[i], all_pred[i])
    ord_pred.append(all_pred[i])

final_table['goswami'] = ord_pred

final_table.to_csv(r'../mdata/final.csv', index=False)