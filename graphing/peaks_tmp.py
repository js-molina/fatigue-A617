import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib, os, sys
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.serif'] = 'Computer Modern'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'

sys.path.append('/../../')

sys.path.append('..')

from fatigue.graph import *

from temp.get_folds import Data

# fig, ax = plt.subplots(1, 1)

# cp0 = ax.scatter(np.ones(len(Data))*(-500), np.zeros(len(Data)), c = Data.Cycles.sort_values(), cmap='coolwarm')
# cb = fig.colorbar(cp0, ax=ax) 

# plt.show()

# #%%

# fig, ax = plt.subplots(1, 2, sharey = True, figsize=(10,4))
# fig.subplots_adjust(wspace = 0.15)

# kwargs = [{'temp' : 850}, {'temp' : 950}]

# for i in range(2):
#     tmp = Data.sort_values(by='Cycles', ignore_index = True)

#     if 'temp' in kwargs[i]:
#         tmp = tmp.loc[tmp.Temps == kwargs[i]['temp']]
#     if 'strain' in kwargs[i]:
#         tmp = tmp.loc[tmp.Strains == kwargs[i]['strain']]        
#     if 'rate' in kwargs[i]:
#         tmp = tmp.loc[tmp.Rates == kwargs[i]['rate']]  

#     colors = cp0.get_facecolors()[tmp.index]
    
#     for j, sample in enumerate(tmp.Samples):
#         test, = fatigue_data.get_test_from_sample(sample)
#         df = get_peak_data_from_test(test)
        
#         ax[i].plot(df.Cycle, df['Max Stress Mpa'], lw = 0.8, color = colors[j])
#         ax[i].plot(df.Cycle, df['Min Stress Mpa'], lw = 0.8, color = colors[j])
    
     
#     ax[i].set_xlabel("Cycles")
#     ax[0].set_ylabel("Max/Min Stress (MPa)", labelpad = 5)
    
#     ax[i].set_xscale('log')
    
#     ax[i].grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
#     ax[i].set_xlim([10, 20000])
#     ax[i].set_ylim(-350, 350)

#     # ax[i].xaxis.set_minor_locator(MultipleLocator(500))

#     ax[i].yaxis.set_minor_locator(MultipleLocator(20))

# ax[0].tick_params(axis = 'both', direction='inout', which = 'both')
# ax[1].tick_params(axis = 'y', direction='inout', which = 'both')

# cb = fig.colorbar(cp0, ax=ax) 
# cb.set_label(label = 'Cycles $N_f$', labelpad = 5)

# path = r'D:\WSL\ansto\figs'
# # plt.savefig(os.path.join(path, 'all_peaks2.svg'), bbox_inches = 'tight')

# plt.show()


tmp = Data.sort_values(by='Cycles')
tmp950 = tmp[tmp.Temps == 950]
tmp850 = tmp[tmp.Temps == 850]

samples950 = []
samples850 = []

for st in tmp850.Strains.unique():
    _ = tmp850[tmp850.Strains == st]
    samples850.append(_.Samples.tolist())

for st in tmp950.Strains.unique():
    _ = tmp950[tmp950.Strains == st]
    samples950.append(_.Samples.tolist())

samples950 = reversed(samples950)
samples850 = reversed(samples850)

# samples950 = reversed(['J4', 'E12', 'B14', '4316', 'B3'])
# samples850 = reversed(['41614', '41621', '4169', '4165', '411', '436'])

strains = list(reversed([3.0 , 2.0 , 1.0, 0.6, 0.4, 0.3]))
w = [0.9, 1, 1.2, 1.8, 2, 2.4]
w = [0.8]*6

samples = [samples850, samples950]

fig = plt.figure(figsize=(7,3))
ax = [fig.add_axes([0, 0, 3/7, 1]), fig.add_axes([3.8/7, 0, 3/7, 1])]

colors = list(reversed(['red', 'xkcd:orange', '#ff66ff', '#9933ff', 'xkcd:green', '#0099ff']))

t = [850, 950]

for i in range(2):
    
    for j, sample_ in enumerate(samples[i]):
        for k, sample1 in enumerate(sample_):
            test, = fatigue_data.get_test_from_sample(sample1)
            df = get_peak_data_from_test(test)
            if k == 0:
                ax[i].plot(df.Cycle, df['Max Stress Mpa'], lw = w[j], color = colors[j], label = f'{strains[j]:.1f}\%')
            else:
                ax[i].plot(df.Cycle, df['Max Stress Mpa'], lw = w[j], color = colors[j])
            ax[i].plot(df.Cycle, df['Min Stress Mpa'], lw = w[j], color = colors[j])
        
    ax[i].set_xlim([10, 2e4])
    ax[i].set_ylim(-350, 350)
    
    ax[i].set_xscale('log')
    
    ax[i].set_xlabel("Cycles")
    ax[i].set_ylabel("Max/Min Stress (MPa)", labelpad = 5)
    ax[i].yaxis.set_minor_locator(MultipleLocator(20))    
    
    ax[i].set_title(r'\SI{%d}{\celsius}'%t[i])
    
    ax[i].grid(dashes = (1, 5), color = 'gray', lw = 0.7)

    ax[i].tick_params(axis = 'y', direction='inout', which = 'major', length = 4)
    ax[i].tick_params(axis = 'y', direction='inout', which = 'minor', length = 2)

# ax[i].legend(facecolor = 'white', edgecolor = 'none', ncol = 1, loc = 'lower right', title = 'Strain Range')

handles, labels = ax[0].get_legend_handles_labels()
lgd = fig.legend(handles, labels, ncol = 6, title = 'Strain Range', facecolor = 'white', edgecolor = 'none', \
            framealpha = 1, bbox_to_anchor=(0.97, 1.3), fontsize=12)
    
# plt.setp(lgd.get_title(), multialignment='center')

for line in lgd.get_lines():
    line.set_linewidth(3.0)

path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'

# path = r'D:\INDEX\Notes\Semester_14\MMAN9451\Thesis A\figs'

# plt.savefig(os.path.join(path, 'strain_peaks.pdf'), bbox_inches = 'tight')

plt.show()

