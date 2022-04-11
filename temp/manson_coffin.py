import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib, os, sys

sys.path.append('..')

from get_folds import Data
from fatigue.finder import fatigue_data
from fatigue.finder.cycle_path import cycle_path_from_sample
from fatigue.strain import cycle_plastic_strain_percent, cycle_elastic_strain_percent

from sklearn.metrics import r2_score

matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{siunitx,amsmath, amssymb, amsfonts, amsthm}'

TEMPS = [850, 950]

    
half_live_labels = ['46%', '47%', '48%', '49%', '50%', '51%', '52%', '53%', '54%', '55%', '56%', '57%']

def lin_to_log(m, a):
    return lambda x: np.exp(a)*(2*x)**m

def manson_coffin(mp, ap, me, ae):
    # print(np.exp(ap), mp, np.exp(ae), me)
    return lambda x: np.exp(ap)*(2*x)**mp + np.exp(ae)*(2*x)**me 


fig = plt.figure(figsize=(7,3))
ax = [fig.add_axes([0, 0, 3/7, 1]), fig.add_axes([3.8/7, 0, 3/7, 1])]

for i, t in enumerate(TEMPS):
    DATA = Data[Data.Temps == t]
    half_live_cycles = []

    for sample in DATA.Samples:
        tmp = pd.read_csv(cycle_path_from_sample(sample))
        cycles = tmp['Cycle Label'].unique()
        for l in half_live_labels:
            for c in cycles:
                if c.startswith(l):
                    half_live_cycles.append((sample, tmp.loc[tmp['Cycle Label'] == c]))
    
    elastic_strain = []
    plastic_strain = []
    total_strain = []
    cycles_to_failure = []
    
    for sample, half_cycle in half_live_cycles:
        elastic_strain.append(cycle_elastic_strain_percent(half_cycle, 950)/100)
        plastic_strain.append(cycle_plastic_strain_percent(half_cycle)/100)
        total_strain.append(cycle_plastic_strain_percent(half_cycle)/100+cycle_elastic_strain_percent(half_cycle, 950)/100)
        # total_strain.append(float(DATA[DATA.Samples == sample].Strains/100))
        cycles_to_failure.append(int(DATA[DATA.Samples == sample].Cycles))
        
    plog = np.polyfit(np.log(2*np.array(cycles_to_failure)), np.log(plastic_strain), 1)
    elog = np.polyfit(np.log(2*np.array(cycles_to_failure)), np.log(elastic_strain), 1)
    
    print(r2_score(np.log(elastic_strain), np.poly1d(elog)(np.log(2*np.array(cycles_to_failure)))))
    
    p = lin_to_log(*plog)
    e = lin_to_log(*elog)
    mc = manson_coffin(*plog, *elog)
    xlim = np.array([min(cycles_to_failure), max(cycles_to_failure)])
    
    ax[i].plot(cycles_to_failure, plastic_strain, 'bs', markerfacecolor = 'None', markeredgewidth = 1)
    ax[i].plot(cycles_to_failure, elastic_strain, 'r^', markerfacecolor = 'None', markeredgewidth = 1)
    ax[i].plot(cycles_to_failure, total_strain, 'go', markerfacecolor = 'None', markeredgewidth = 1)

    ax[i].plot(-1, -1, 'bs-', markerfacecolor = 'None', markeredgewidth = 1, lw = 1, label = 'Plastic Strain')
    ax[i].plot(-1, -1, 'r^-', markerfacecolor = 'None', markeredgewidth = 1, lw = 1, label = 'Elastic Strain')
    ax[i].plot(-1, -1, 'go-', markerfacecolor = 'None', markeredgewidth = 1, lw = 1, label = 'Total Strain')

    x = np.logspace(*np.log10(xlim))

    ax[i].plot(xlim, p(xlim), 'b-', lw = 1)
    ax[i].plot(xlim, e(xlim), 'r-', lw = 1)
    ax[i].plot(x, mc(x), 'g-', lw = 1)

    ax[i].set_title(r'\SI{%d}{\celsius}'%t)

    ax[i].legend(facecolor = 'white', edgecolor = 'none')

    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    
    ax[i].set_xlim(1e1, 1e5)
    ax[i].set_ylim(1e-4, 1e-1)
    
    ax[i].set_ylabel(r'Strain Amplitude, $\Delta\varepsilon$')
    ax[i].set_xlabel(r'Cycles to Failure, $N_f$')

    ax[i].grid(dashes = (1, 5), color = 'gray', lw = 0.7)

path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'

# plt.savefig(os.path.join(path, 'coffman.svg'), bbox_inches = 'tight')

plt.show()

    # print(cycles)