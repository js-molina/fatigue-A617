import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib, os, sys

sys.path.append('..')

from scipy.optimize import root, fsolve
from get_folds import Data
from fatigue.finder import fatigue_data
from fatigue.finder.cycle_path import peak_path_from_sample
from fatigue.strain import cycle_plastic_strain_percent, cycle_elastic_strain_percent
from fatigue.graph.models2 import graph_nn_pred_all, graph_nn_1_fold, graph_nn_2_fold, graph_nn_hist, graph_nn_11_dev, \
     graph_nn_12_dev, get_chi, get_meap
import tdt

from itertools import groupby

from sklearn.metrics import r2_score

fold = 'best'
train, dev, test = tdt.train_idx[fold], tdt.dev_idx[fold], tdt.test_idx[fold]

matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{siunitx,amsmath, amssymb, amsfonts, amsthm}'

TEMPS = [850, 950, 'a']
    
half_live_labels = ['46%', '47%', '48%', '49%', '50%', '51%', '52%', '53%', '54%', '55%', '56%', '57%']

def lin_to_log(m, a):
    return lambda x: np.exp(a)*(2*x)**m

def manson_coffin(mp, ap, me, ae):
    # print(np.exp(ap), mp, np.exp(ae), me)
    return lambda x: np.exp(ap)*(2*x)**mp + np.exp(ae)*(2*x)**me

def manson_coffin_grad(mp, ap, me, ae):
    # print(np.exp(ap), mp, np.exp(ae), me)
    return lambda x: np.exp(ap)*mp*2*(2*x)**(mp-1) + np.exp(ae)*me*2*(2*x)**(me-1)

train_dev = np.concatenate((train, dev))

D_train = Data.loc[train_dev]
D_test = Data.loc[test]

mcs = []; mcsd = []

E = [153e3, 144e3, 148.5e3]

fig = plt.figure(figsize=(7,3))
ax = [fig.add_axes([0, 0, 3/7, 1]), fig.add_axes([3.8/7, 0, 3/7, 1])]

for i, t in enumerate(TEMPS):

    DATA = Data[Data.Temps == t] if t != 'a' else Data
    
    elastic_strain = []
    plastic_strain = []
    total_strain = []
    cycles_to_failure = []
    
    elastic_strain_test = []
    plastic_strain_test = []
    total_strain_test = []
    cycles_to_failure_test = []
    
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
            elastic_strain.append(elastic)
            plastic_strain.append(plastic)
            total_strain.append(total)
            cycles_to_failure.append(cycles)
        else:
            elastic_strain_test.append(elastic)
            plastic_strain_test.append(plastic)
            total_strain_test.append(total)
            cycles_to_failure_test.append(cycles)
    
    elastic_strain, plastic_strain, total_strain, cycles_to_failure = \
    map(np.array, [elastic_strain, plastic_strain, total_strain, cycles_to_failure])

    elastic_strain_test, plastic_strain_test, total_strain_test, cycles_to_failure_test = \
    map(np.array, [elastic_strain_test, plastic_strain_test, total_strain_test, cycles_to_failure_test])       
    
    plog = np.polyfit(np.log(2*cycles_to_failure), np.log(plastic_strain/2), 1)
    elog = np.polyfit(np.log(2*cycles_to_failure), np.log(elastic_strain/2), 1)
    
    print(r2_score(np.log(elastic_strain), np.poly1d(elog)(np.log(2*np.array(cycles_to_failure)))))
    
    p = lin_to_log(*plog)
    e = lin_to_log(*elog)
    mc = manson_coffin(*plog, *elog)
    mcs.append(mc)
    mcsd.append(manson_coffin_grad(*plog, *elog))
    
    xlim = np.array([1e2/2, 1e5])
    
    if i == 2:
        break
    
    ax[i].plot(2*cycles_to_failure, plastic_strain/2, 'bs', markerfacecolor = 'None', markeredgewidth = 1)
    ax[i].plot(2*cycles_to_failure, elastic_strain/2, 'r^', markerfacecolor = 'None', markeredgewidth = 1)
    ax[i].plot(2*cycles_to_failure, total_strain/2, 'go', markerfacecolor = 'None', markeredgewidth = 1)

    ax[i].plot(-1, -1, 'bs-', markerfacecolor = 'None', markeredgewidth = 1, lw = 1, label = 'Plastic Strain')
    ax[i].plot(-1, -1, 'r^-', markerfacecolor = 'None', markeredgewidth = 1, lw = 1, label = 'Elastic Strain')
    ax[i].plot(-1, -1, 'go-', markerfacecolor = 'None', markeredgewidth = 1, lw = 1, label = 'Total Strain')

    x = np.logspace(*np.log10(xlim))

    ax[i].plot(2*xlim, p(xlim), 'b-', lw = 1)
    ax[i].plot(2*xlim, e(xlim), 'r-', lw = 1)
    ax[i].plot(2*x, mc(x), 'g-', lw = 1)

    if i == 2:
        ax[i].set_title('Hybrid')
    else:
        ax[i].set_title(r'\SI{%d}{\celsius}'%t)

    ax[i].legend(facecolor = 'white', edgecolor = 'k', framealpha = 1)

    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    
    ax[i].set_xlim(1e2, 1e5)
    ax[i].set_ylim(1e-4, 1e-1)
    
    ax[i].set_ylabel(r'Strain Amplitude, $(\Delta\varepsilon/2)$')
    ax[i].set_xlabel(r'Reversals to Failure, $(2N_f)$')

    ax[i].grid('-', color = 'gray', lw = 0.7, which = 'major')
    ax[i].grid(ls = ':', color = 'gray', lw = 0.5, which = 'minor')

path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'
path = r'D:\INDEX\Notes\Semester_16\MMAN4953\Thesis C\img'

# plt.savefig(os.path.join(path, 'mc_all.pdf'), bbox_inches = 'tight')

#%%

strains = Data.Strains/100
obs = Data.Cycles

pred = []

# for i, x in enumerate(strains):
#     if Data.Temps[i] == 850:
#         tmp = Data[(Data.Temps == 850) & (Data.Strains == x*100)]
#         fun = lambda t, x = x: mcs[0](t) - x
#         grad = mcsd[0]
#         x0 = tmp.Cycles.mean()
#     else:
#         tmp = Data[(Data.Temps == 950) & (Data.Strains == x*100)]
#         fun = lambda y, x = x: mcs[1](y) - x
#         grad = mcsd[1]
#         x0 = tmp.Cycles.mean()
    
#     sol = fsolve(fun, x0 = x0, fprime = grad)
    
#     pred.append(*sol)

for i, x in enumerate(strains):
    np.random.seed(123)
    if Data.Temps[i] == 850:
        tmp = Data[(Data.Temps == 850) & (Data.Strains == x*100)]
        fun = lambda t, x = x: mcs[0](t) - x
        grad = mcsd[0]
        x0 = tmp.Cycles.sample()
    else:
        tmp = Data[(Data.Temps == 950) & (Data.Strains == x*100)]
        fun = lambda y, x = x: mcs[1](y) - x
        grad = mcsd[1]
        x0 = tmp.Cycles.sample()
    
    sol = fsolve(fun, x0 = x0, fprime = grad)
    
    pred.append(*sol)
    
obs = obs.to_numpy()
pred = np.array(pred)

op850 = np.array([(i, j) for i, j in zip(obs, pred) if i in D_train[D_train.Temps == 850].Cycles.to_list()])
op950 = np.array([(i, j) for i, j in zip(obs, pred) if i in D_train[D_train.Temps == 950].Cycles.to_list()])

op850t = np.array([(i, j) for i, j in zip(obs, pred) if i in D_test[D_test.Temps == 850].Cycles.to_list()])
op950t = np.array([(i, j) for i, j in zip(obs, pred) if i in D_test[D_test.Temps == 950].Cycles.to_list()])


#%%

obs = Data.Cycles

pred = []

for i, x in enumerate(strains):
    np.random.seed(123)
    tmp = Data[(Data.Strains == x*100)]
    fun = lambda t, x = x: mcs[2](t) - x
    grad = mcsd[2]
    x0 = tmp.Cycles.sample()

    sol = fsolve(fun, x0 = x0, fprime = grad)
    
    pred.append(*sol)
    
obs = obs.to_numpy()
pred = np.array(pred)

#%%

fig, ax = plt.subplots()

ax.set_aspect('equal')

ax.set_ylim(100, 20000)
ax.set_xlim(100, 20000)

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlabel('Predicted $N_f$')
ax.set_ylabel('Measured $N_f$')

ax.plot([50, 20000], [50, 20000], lw = 2, color = 'k')
ax.fill_between([50, 20000], 50, [50, 20000], color = 'k', alpha = 0.1)

ax.plot([100, 1e5], [200, 2e5], lw = 1, ls = '--', color = 'gray')
ax.plot([200, 2e5], [100, 1e5], lw = 1, ls = '--', color = 'gray')

ax.plot(op850[:, 1], op850[:, 0], 'bo', markersize = 7, ls = 'None', markerfacecolor = 'None', \
        markeredgewidth = 2, label = r'\SI{850}{\celsius} -- Train, (%.1f'%((abs(op850[:, 0]-op850[:, 1])/op850[:, 0]).mean()*100) + '\%)')
ax.plot(op850t[:, 1], op850t[:, 0], 'bx', markersize = 7, ls = 'None', markerfacecolor = 'None', \
        markeredgewidth = 2, label = r'\SI{850}{\celsius} -- Test, (%.1f'%((abs(op850t[:, 0]-op850t[:, 1])/op850t[:, 0]).mean()*100) + '\%)')
    
    
ax.plot(op950[:, 1], op950[:, 0], 'ro', markersize = 7, ls = 'None', markerfacecolor = 'None', \
        markeredgewidth = 2, label = r'\SI{950}{\celsius} -- Train, (%.1f'%((abs(op950[:, 0]-op950[:, 1])/op950[:, 0]).mean()*100) + '\%)')
ax.plot(op950t[:, 1], op950t[:, 0], 'rx', markersize = 7, ls = 'None', markerfacecolor = 'None', \
        markeredgewidth = 2, label = r'\SI{950}{\celsius} -- Test, (%.1f'%((abs(op950t[:, 0]-op950t[:, 1])/op950t[:, 0]).mean()*100) + '\%)')
# ax.plot(pred, obs, 's', markersize = 7, ls = 'None', markerfacecolor = 'None', markeredgecolor = '#ff9900', \
#     markeredgewidth = 2, label = r'Hybrid -- $%.1f$'%((abs(obs-pred)/obs).mean()*100) + '\%')

ax.tick_params(axis = 'both', direction='in', which = 'both')
ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)

props = dict(boxstyle='round', facecolor= (0.9, 0.9, 0.9), lw = 1)
ax.text(0.95, 0.05, r'\textit{Non-Conservative Region}', transform=ax.transAxes, va='bottom', ha = 'right', bbox=props)
ax.set_title(r'\textbf{Coffin-Manson}')

legend = ax.legend(framealpha = 1, edgecolor = 'k', loc = 2)
# for t in legend.get_texts():
    # t.set_ha('right')
    # t.set_fontsize(9)
path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'

np.savez('../mdata/cm850tt', y_pred_train = op850[:, 1], y_obs_train = op850[:, 0],
                            y_pred_test = op850t[:, 1], y_obs_test = op850t[:, 0])
np.savez('../mdata/cm950tt', y_pred_train = op950[:, 1], y_obs_train = op950[:, 0],
                            y_pred_test = op950t[:, 1], y_obs_test = op950t[:, 0])

# plt.savefig(os.path.join(path, 'cmanson_split.pdf'), bbox_inches = 'tight')
plt.show()  

#%%
final_table = pd.read_csv(r'../mdata/final.csv')

all_obs = list(np.concatenate((op850[:, 0], op850t[:, 0], op950[:, 0], op950t[:, 0])).astype('int64'))
all_pred = np.concatenate((op850[:, 1], op850t[:, 1], op950[:, 1], op950t[:, 1]))


ord_pred = []

for c in Data.Cycles:
    i = all_obs.index(c)
    print(all_obs[i], all_pred[i])
    ord_pred.append(all_pred[i])

final_table['coffin_manson'] = ord_pred

final_table.to_csv(r'../mdata/final.csv')
