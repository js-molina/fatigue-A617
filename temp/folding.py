import matplotlib.pyplot as plt
import matplotlib, os
import numpy as np

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.serif'] = 'Computer Modern'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'

log = True

fig, taxes = plt.subplots(3, 3, sharex=True, sharey = True, figsize=(9,9))
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel("Observed $N_f$", fontsize=13, )
plt.xlabel("Predicted $N_f$", fontsize=13)

axes = [taxes[i][j] for i in range(3) for j in range(3)]

colors = ['b', 'r']
markers = ['x', 'o']
labels = []

for i in range(9):
    j = i+1
    
    d = np.load('../mdata/break/1_%d.npz'%j)
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
    
    axes[i].plot(x0, y0, color = colors[0], ls = 'none', marker = markers[0], markersize = 6, \
              markerfacecolor = 'None', label = 'Train Data')
    axes[i].plot(x1, y1, color = colors[1], ls = 'none', marker = markers[1], markersize = 6, \
              markerfacecolor = 'None', label = 'Test Data')
    
    axes[i].set_title('Fold %d'%j)

handles, labels = axes[i].get_legend_handles_labels()
lgd = fig.legend(handles, labels, ncol = 2, facecolor = 'white', edgecolor = 'none', \
            framealpha = 0, bbox_to_anchor=(0.7, 0.96), fontsize=12)

# path = r'D:\WSL\ansto\figs'

# plt.savefig(os.path.join(path, 'folds.pdf'))
    
    
plt.show()
