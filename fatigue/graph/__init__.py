import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib, os, sys

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.serif'] = 'Computer Modern'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'

from .funcs import *
from .helper import *
from .models import *
from .validation import *
from ..finder.cycle_path import *

sys.path.append('/../../')

from temp.get_folds import Data

main_path = os.path.abspath(__file__ + "/../")

fig_path = os.path.join(main_path, 'figs')

def graph_cycles_from_test(test, name = ''):

    cycles = get_cycles_from_test(test)

    ax = plt.gca()
    
    ax.set_xlabel("Strain (mm/mm)")
    ax.set_ylabel("Stress (MPa)")
    
    ax.set_title('Sample {%s} -- \SI{%d}{\celsius}, %.2f'% \
             (test.Sample, test.Temp, test.Strain) + '\% Strain', size = 11)
    
    colors = plt.cm.viridis(np.linspace(0,1,len(cycles)))
    
    for i, c in enumerate(cycles):
        graph_cycle(c, colors[i], ax)
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    if name:
        plt.savefig(os.path.join(fig_path, name + '.pdf'), bbox_inches = 'tight')

    plt.show()

    print('%d cycles'%len(cycles))

    
def graph_peaks_from_test(test, name = ''):

    df = get_peak_data_from_test(test)

    ax = plt.gca()

    mm = max(max(df['Max Stress Mpa']), abs(min(df['Min Stress Mpa'])))

    ax.set_ylim([-100*np.ceil(mm/100), 100*np.ceil(mm/100)])
    ax.set_xlim([0, len(df)+10])

    ax.set_title('Sample {%s} -- \SI{%d}{\celsius}, %.2f'% \
                 (test.Sample, test.Temp, test.Strain) + '\% Strain', size = 11)

    ax.set_xlabel("Cycles")
    ax.set_ylabel("Max/Min Stress (MPa)")

    ax.plot(df.Cycle, df['Max Stress Mpa'], lw = 0.8, color = 'red')
    ax.plot(df.Cycle, df['Min Stress Mpa'], lw = 0.8, color = 'blue')

    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)

    if name:
        plt.savefig(os.path.join(fig_path, name + '.pdf'), bbox_inches = 'tight')
    
    plt.show()

def graph_all_peaks(save = '', **kwargs):
    
    tmp = Data.sort_values(by='Cycles', ignore_index=True)
    
    if 'temp' in kwargs:
        tmp = tmp.loc[tmp.Temps == kwargs['temp']]
    if 'strain' in kwargs:
        tmp = tmp.loc[tmp.Strains == kwargs['strain']]        
    if 'rate' in kwargs:
        tmp = tmp.loc[tmp.Rates == kwargs['rate']]      
    
    fig, ax = plt.subplots(1, 1)
    
    cp = ax.scatter(np.ones(len(tmp))*(-500), np.zeros(len(tmp)), c = tmp.Cycles, cmap='coolwarm')
    cb = fig.colorbar(cp, ax=ax) 
    
    plt.show()
    plt.clf()
    
    colors = cp.get_facecolors()
    
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    
    # cols = plt.cm('coolwarm', tmp.Cycles/tmp.Cycles.max())
    # norm = matplotlib.colors.Normalize(vmin = tmp.Cycles.min(), vmax = tmp.Cycles.max())
    # colors = plt.cm.ScalarMappable(cmap = plt.get_cmap('coolwarm', len(tmp)), norm = norm).get_cmap()
    
    for i, sample in enumerate(tmp.Samples):
        test, = fatigue_data.get_test_from_sample(sample)
        df = get_peak_data_from_test(test)
        
        mm = max(max(df['Max Stress Mpa']), abs(min(df['Min Stress Mpa'])))
    
        # ax.set_ylim([-100*np.ceil(mm/100), 100*np.ceil(mm/100)])
        
        ax.set_xlabel("Cycles")
        ax.set_ylabel("Max/Min Stress (MPa)")
    
        ax.plot(df.Cycle, df['Max Stress Mpa'], lw = 0.8, color = colors[i])
        ax.plot(df.Cycle, df['Min Stress Mpa'], lw = 0.8, color = colors[i])
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    ax.set_xlim([0, 500*np.ceil(tmp.Cycles.max()/500)])
    
    if 'xlim' in kwargs:
        ax.set_xlim([0, kwargs['xlim']])
    
    cp = ax.scatter(np.ones(len(tmp))*(-500), np.zeros(len(tmp)), c = tmp.Cycles, cmap='coolwarm')
    cb = fig.colorbar(cp, ax=ax) 
    cb.set_label(label = 'Cycles $N_f$', labelpad = 10)
    
    ax.set_ylim(-350, 350)
    
    if save:
        path = r'D:\WSL\ansto\figs'
        plt.savefig(os.path.join(path, save + '.svg'), bbox_inches = 'tight')
    
    plt.show()
    
    return(colors)
    