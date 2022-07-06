import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)

sys.path.append('..')

import sklearn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from fatigue.finder import fatigue_data, fd_to_df
from fatigue.graph.models2 import graph_nn_pred_all, graph_nn_pred_strain, graph_nn_prediction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from fatigue.models2.helper import get_nf

test_idx = {}
dev_idx = {}
train_idx = {}

plot = False

# np.random.seed(19)

def f(x, S):
    return len(S[S <= x])/len(S)

Data = fd_to_df(fatigue_data.data).sort_values(by=['Temps', 'Strains'])

cycles = [get_nf(sample, from_sample=True) for sample in Data.Samples]

Data = Data.assign(Cycles = cycles)

Data_850, Data_950 = Data[Data.Temps == 850], Data[Data.Temps == 950]

test_samples = []
dev_samples = []
for st in Data_850.Strains.unique():
    tmp = Data_850[Data_850.Strains == st]
    test_samples.append(tmp.Samples.sample().iloc[0])
    tmp = tmp[~tmp.Samples.isin(test_samples)]
    dev_samples.append(tmp.Samples.sample().iloc[0])

for st in Data_950.Strains.unique():
    tmp = Data_950[Data_950.Strains == st]
    test_samples.append(tmp.Samples.sample().iloc[0])
    tmp = tmp[~tmp.Samples.isin(test_samples)]
    dev_samples.append(tmp.Samples.sample().iloc[0])
    
test_samples = ['4322',  '41615',  '4165',  '4167',  '41611',  '41614',  'B1',  '4316',  'B13',  'E11',  '4317']
dev_samples = ['4313',  '41620',  '4168',  '4169',  '41610',  '41619',  'B3',  '4320',  'B14',  '433',  'J5']

Data_test = Data[Data.Samples.isin(test_samples)].sort_values(by=['Cycles'])
Data_dev = Data[Data.Samples.isin(dev_samples)].sort_values(by=['Cycles'])
Data_train = Data[~(Data.Samples.isin(test_samples) | Data.Samples.isin(dev_samples))].sort_values(by='Cycles')

cTest = Data_test.Cycles
cDev = Data_dev.Cycles
cTrain = Data_train.Cycles

#%%

if plot:
    print('Summary Statistics:\tMean\tMedian\tStd\t\tMin\t\tMax')
    print(f'Test Data:\t\t\t{cTest.mean():.1f}\t{cTest.median():.1f}\t{cTest.std():.1f}\t{cTest.min():.1f}\t{cTest.max():.1f}')
    print(f'Dev Data:\t\t\t{cDev.mean():.1f}\t{cDev.median():.1f}\t{cDev.std():.1f}\t{cDev.min():.1f}\t{cDev.max():.1f}')
    print(f'Train Data:\t\t\t{cTrain.mean():.1f}\t{cTrain.median():.1f}\t{cTrain.std():.1f}\t{cTrain.min():.1f}\t{cTrain.max():.1f}')
    # plot_test_train(cTrain, cTest, save = 'best_data.svg')
    # plot_cum_prob(cTrain, cTest, save = 'best_prob.svg')
    fig = plt.figure(figsize=(7,3))
    ax1 = fig.add_axes([0, 0, 3/7, 1])
    ax2 = fig.add_axes([3.8/7, 0, 3/7, 1])
    
    ax1.set_xlabel('$N_f$ - Observed')

    ax1.set_ylim(0, 4)
    ax1.set_xlim(100, 20000)
    ax1.set_xscale('log')
    
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(['Training', 'Development', 'Testing'], rotation = 90, va = 'center')
    
    ax1.plot(cTrain, np.ones(cTrain.shape), marker = 'o', markersize = 5, ls = 'None', \
    color = 'blue', label = 'Training')
    
    rated = {0.001 : '0.001', 0.0001 : '0.0001', 0.00001 : '0.00001'}
    
    cols = ['blue', 'xkcd:green', 'red']
    dsets = [cTrain, cDev, cTest]
    
    for j, dset in enumerate(dsets):
        
        sp = Data.loc[dset.index]#.sample(5).sort_values(by = 'Cycles')
    
        for i, e in enumerate(sp.itertuples()):
            # ax1.annotate(r'(%d, %.1f, %s)'%(e.Temps, e.Strains, rated[e.Rates]),
            #              xy=(e.Cycles, 1), xytext=(e.Cycles, j+1.1), ha = 'left', rotation = 45, fontsize=7,
            #              color = cols[j])
            # ax1.annotate(r'$(\SI{%d}{\celsius}, %.1f\%%)$'%(e.Temps, e.Strains),
            #              xy=(e.Cycles, 1), xytext=(e.Cycles, j+1.1), ha = 'center', rotation = 90, fontsize=7,
            #              color = cols[j])
            ax1.annotate(r'(%d, %.1f)'%(e.Temps, e.Strains),
                          xy=(e.Cycles, 1), xytext=(e.Cycles*1.035, j+1.1), ha = 'center', rotation = 90, fontsize=6.5,
                          color = cols[j])
        
    ax1.plot(cDev, 2*np.ones(cDev.shape), marker = 'o', markersize = 5, ls = 'None', \
    color = 'xkcd:green', label = 'Development')  
    
    ax1.plot(cTest, 3*np.ones(cTest.shape), marker = 'o', markersize = 5, ls = 'None', \
    color = 'red', label = 'Testing')
    
    ax2.set_xlabel('$N_f$ - Observed')
    ax2.set_ylabel('Cumulative Probability')
    
    ax2.set_ylim(0, 1)
    ax2.set_xlim(100, 20000)
    ax2.set_xscale('log')
    
    t_train = [100] + cTrain.tolist() + [20000]
    t_dev = [100] + cDev.tolist() + [20000]
    t_test = [100] + cTest.tolist() + [20000]
    
    ax2.step(t_train, [f(x, cTrain) for x in t_train], where = 'post', lw = 0.8, color = 'blue', label = 'Training')
    ax2.step(t_dev, [f(x, cDev) for x in t_dev], where = 'post', lw = 0.8, color = 'xkcd:green', label = 'Development')
    ax2.step(t_test, [f(x, cTest) for x in t_test], where = 'post', lw = 0.8, color = 'red', label = 'Testing')  
    
    ax2.legend(framealpha = 1, edgecolor = 'None')
    
    ax1.grid(color = '#f2f2f2', lw = 0.5, which = 'both')
    ax2.grid(color = '#f2f2f2', lw = 0.5)
    
    path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'
    plt.savefig(os.path.join(path, 'splits.pdf'), bbox_inches = 'tight')
    
    plt.show()

test_idx['best'] = cTest.index
dev_idx['best'] = cDev.index
train_idx['best'] = cTrain.index
