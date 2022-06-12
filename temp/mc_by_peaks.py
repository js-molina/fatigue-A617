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

fold = 'best'

train, dev, test = tdt.train_idx[fold], tdt.dev_idx[fold], tdt.test_idx[fold]

from sklearn.metrics import r2_score

matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{siunitx,amsmath, amssymb, amsfonts, amsthm}'

TEMPS = [850, 950]
    
half_live_labels = ['46%', '47%', '48%', '49%', '50%', '51%', '52%', '53%', '54%', '55%', '56%', '57%']

def lin_to_log(m, a):
    return lambda x: np.exp(a)*(2*x)**m

def manson_coffin(mp, ap, me, ae):
    # print(np.exp(ap), mp, np.exp(ae), me)
    return lambda x: np.exp(ap)*(2*x)**mp + np.exp(ae)*(2*x)**me

def manson_coffin_grad(mp, ap, me, ae):
    # print(np.exp(ap), mp, np.exp(ae), me)
    return lambda x: np.exp(ap)*mp*2*(2*x)**(mp-1) + np.exp(ae)*me*2*(2*x)**(me-1)


fig = plt.figure(figsize=(7,3))
ax = [fig.add_axes([0, 0, 3/7, 1]), fig.add_axes([3.8/7, 0, 3/7, 1])]
mcs = []; mcsd = []

E = [153e3, 144e3]

D_train = Data.iloc[train]
D_dev = Data.iloc[dev]
D_test = Data.iloc[test]

for i, t in enumerate(TEMPS):
    DATA = Data[Data.Temps == t]
    half_live_cycles = []

    elastic_strain = []
    plastic_strain = []
    total_strain = []
    cycles_to_failure = []
    
    elastic_strain_model = []
    plastic_strain_model = []
    total_strain_model = []
    cycles_to_failure_model = []

    for sample in DATA.Samples:
        tmp = pd.read_csv(peak_path_from_sample(sample))
        tx = tmp.iloc[tmp.Cycle.iloc[-1]//2]
        max_s, min_s =  map(float, tx[['Max Stress Mpa', 'Min Stress Mpa']])
        s_ratio = abs(max_s/min_s)
        total = float(DATA[DATA.Samples == sample].Strains/100)
        elastic = (max_s-min_s)/E[i]
        plastic = total - elastic
        cycles = int(DATA[DATA.Samples == sample].Cycles)
    
        elastic_strain.append(elastic)
        plastic_strain.append(plastic)
        total_strain.append(total)
        cycles_to_failure.append(cycles)
        
        if sample in D_train.Samples.values:
            elastic_strain_model.append(elastic)
            plastic_strain_model.append(plastic)
            total_strain_model.append(total)
            cycles_to_failure_model.append(cycles)
        
    plog = np.polyfit(np.log(2*np.array(cycles_to_failure_model)), np.log(plastic_strain_model), 1)
    elog = np.polyfit(np.log(2*np.array(cycles_to_failure_model)), np.log(elastic_strain_model), 1)
    
    print(r2_score(np.log(elastic_strain_model), np.poly1d(elog)(np.log(2*np.array(cycles_to_failure_model)))))
    
    p = lin_to_log(*plog)
    e = lin_to_log(*elog)
    mc = manson_coffin(*plog, *elog)
    mcs.append(mc)
    mcsd.append(manson_coffin_grad(*plog, *elog))
    xlim = np.array([min(cycles_to_failure), max(cycles_to_failure)])
    
    def where_cycle(n):
        if n in D_train.Cycles.values:
            return 'train'
        elif n in D_dev.Cycles.values:
            return 'dev'
        else:
            return 'test'
    
    data = np.array(sorted(zip(cycles_to_failure, elastic_strain, plastic_strain, total_strain), key = lambda x: where_cycle(x[0])))
     
    ctf, el, pl, tl = np.hsplit(data, 4)
    
    i_test = np.array([i in D_test.Cycles.values for i in ctf])
    i_dev = np.array([i in D_dev.Cycles.values for i in ctf])
    i_train = np.array([i in D_train.Cycles.values for i in ctf])
    
    j = {'train' : i_train, 'dev' : i_dev, 'test' : i_test}
    
    for l, (k, g) in enumerate(groupby(ctf, where_cycle)):
        
        ms = 7        
        mc1, mc2, mc3 = 'None', 'None', 'None'
        hatch = ''
        if l == 2:
            mc1, mc2, mc3 = 'brg'
        
        ii = j[k]        
                
        ax[i].plot(ctf[ii], pl[ii], 's', color = 'blue', markerfacecolor = mc1, markersize = ms)
        ax[i].plot(ctf[ii], el[ii], '^', color = 'red', markerfacecolor = mc2, markersize = ms)
        ax[i].plot(ctf[ii], tl[ii], 'o', color = 'green', markerfacecolor = mc3, markersize = ms)

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

path = r'D:\INDEX\Notes\Semester_14\MMAN9451\Thesis A\figs'

# plt.savefig(os.path.join(path, 'coffman.pdf'), bbox_inches = 'tight')

plt.show()

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

r_data = {'y_obs_train': obs[train], 'y_pred_train': pred[train], 
          'y_obs_dev': obs[dev], 'y_pred_dev': pred[dev], 
          'y_obs_test': obs[test], 'y_pred_test': pred[test]}

graph_nn_12_dev(r_data, log = True, load = False, which = 'all')
print(get_meap(r_data, load = False))
print(get_chi(r_data, load = False))

#%%

d = r_data

y_obs = np.concatenate((d['y_obs_train'].reshape(22,-1),
            d['y_obs_dev'].reshape(11,-1),
            d['y_obs_test'].reshape(11,-1)), axis = 0)
y_pred = np.concatenate((d['y_pred_train'].reshape(22,-1),
                         d['y_pred_dev'].reshape(11,-1),
                         d['y_pred_test'].reshape(11,-1)), axis = 0)


err850, err950 = [], []

for i, c in enumerate(y_obs):
    if c in Data[Data.Temps == 850].Cycles.values:
        err850.append(abs(c-y_pred[i])/c)
    else:
        err950.append(abs(c-y_pred[i])/c)
    

err850 = np.array(err850)
err950 = np.array(err950)
