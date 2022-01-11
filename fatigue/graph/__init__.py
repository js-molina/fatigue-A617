import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib, os

# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['font.serif'] = 'Computer Modern'
# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'

from .funcs import *
from .helper import *
from .models import *
from ..finder.cycle_path import *

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
    ax.set_xlim([10, len(df)+10])

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

def graph_all_peaks(data):
    
    ax = plt.gca()
    
    for test in data:
        df = get_peak_data_from_test(test)
        
        mm = max(max(df['Max Stress Mpa']), abs(min(df['Min Stress Mpa'])))
    
        # ax.set_ylim([-100*np.ceil(mm/100), 100*np.ceil(mm/100)])
        
        ax.set_xlabel("Cycles")
        ax.set_ylabel("Max/Min Stress (MPa)")
    
        ax.plot(df.Cycle, df['Max Stress Mpa'], lw = 0.8, color = 'red')
        ax.plot(df.Cycle, df['Min Stress Mpa'], lw = 0.8, color = 'blue')
    
        ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    ax.set_xlim([5, 100])
    
    plt.show()
    