import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import matplotlib, os
import numpy as np
from fatigue.graph.models2 import graph_nn_pred_all, graph_nn_prediction, chi_ratio

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.serif'] = 'Computer Modern'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'

vals = [1, 10, 20, 40, 60, 80, 100, 120, 150, 200, 500, 1000, 5000, 10000, 20000]
n = np.random.choice(vals)
n = 120

print(f'Plotting prediction with {n} cycles!')

log = True

fig, taxes = plt.subplots(2, 2, sharex=False, sharey = False, figsize=(8,8))
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel("Observed $N_f$", fontsize=13, labelpad=12)
plt.xlabel("Predicted $N_f$", fontsize=13, labelpad=12)

axes = [taxes[i][j] for i in range(2) for j in range(2)]

colors = ['b', 'r']
markers = ['x', 'o']
labels = []

for i in range(4):
    j = i+1
    
    d = np.load('../mdata/break/ydata-15-02-22-%d/%d.npz'%(n, j))
    x0, y0, x1, y1 = d['x0'], d['y0'], d['x1'], d['y1']
    
    axes[i].set_ylim(100, 12000)
    axes[i].set_xlim(100, 12000)
    
    if log:
        axes[i].set_ylim(100, 20000)
        axes[i].set_xlim(100, 20000)
        axes[i].set_yscale('log')
        axes[i].set_xscale('log')
    
    
    axes[i].plot([100, 20000], [100, 20000], lw = 2, color = 'k')
    axes[i].fill_between([100, 20000], 100, [100, 20000], color = 'k', alpha = 0.1)
    
    axes[i].plot([100, 20000], [200, 40000], lw = 1, ls = '--', color = 'gray')
    axes[i].plot([200, 40000], [100, 20000], lw = 1, ls = '--', color = 'gray')
    
    axes[i].plot(x0, y0, color = colors[0], ls = 'none', marker = markers[0], markersize = 6, \
              markerfacecolor = 'None', label = 'Train Data')
    axes[i].plot(x1, y1, color = colors[1], ls = 'none', marker = markers[1], markersize = 6, \
              markerfacecolor = 'None', label = 'Test Data')
    
    axes[i].set_title('Fold %d, $\chi^2 = %.2f$'%(j, chi_ratio(np.concatenate((x0, x1), axis = 0), \
                                                               np.concatenate((y0, y1), axis = 0))))
    axes[i].grid(dashes = (1, 5), color = 'gray', lw = 0.7)

handles, labels = axes[i].get_legend_handles_labels()
lgd = fig.legend(handles, labels, ncol = 2, facecolor = 'white', edgecolor = 'none', \
            framealpha = 0, bbox_to_anchor=(0.7, 0.97), fontsize=12)

# path = r'D:\WSL\ansto\figs'
# plt.savefig(os.path.join(path, 'folds.pdf'))
   
plt.show()
#%%

graph_nn_pred_all('../mdata/ydata-15-02-22-%d.npz'%n, log=log, v2 = True)
