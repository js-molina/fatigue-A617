import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib, os, sys

data = ['rlr', 'rnn', 'mc850', 'mc950']
data = ['rlr','rnn', 'gos850', 'gos950']

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

mfc = ['#707070', '#707070', 'b', 'r']
mfc = ['steelblue', 'salmon']
mec = ['k', 'k', 'b', 'r']
# model = 'C-M'
model = 'Goswami'
labels = ['RLR Model', 'RNN Model', r'\SI{850}{\celsius} -- ' + model, r'\SI{950}{\celsius} -- ' + model]
markers=  ['o', 's', 'x', '+']

ms = 8
ew = 1

for i, d in enumerate(data):
    if i >= 2:
        continue
    _ = np.load(f'../mdata/{d}.npz')
    obs, pred = _['y_obs'], _['y_pred']

    if i == 2:
        ms += 3
        ew += 2
    if i == 3:
        ms += 2

    ax.plot(pred, obs, markers[i], markersize = ms,  markerfacecolor = mfc[i], markeredgecolor = mec[i],
            markeredgewidth = ew, label = labels[i])# + ' -- %.2f'%((abs(obs-pred)/obs).mean()*100) + '\%')


ax.tick_params(axis = 'both', direction='in', which = 'both')
ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)

legend = ax.legend(framealpha = 1, edgecolor = 'k', loc = 0)
# for t in legend.get_texts():
    # t.set_ha('right')
    # t.set_fontsize(9)
    

path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'
plt.savefig(os.path.join(path, 'data_only.pdf'), bbox_inches = 'tight')
plt.show()  