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
train_idx = {}

plot = False

def plot_test_train(cTrain, cTest):
    ax = plt.gca()
    
    ax.set_xlabel('$N_f$ - Observed')
    
    ax.set_ylim(0, 3)
    ax.set_xlim(100, 20000)
    ax.set_xscale('log')
    
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['Train', 'Test'])
    
    ax.plot(cTrain, np.ones(cTrain.shape), marker = 'o', markersize = 5, ls = 'None', \
    color = 'blue', label = 'Train')  
    
    ax.plot(cTest, 2*np.ones(cTest.shape), marker = 'o', markersize = 5, ls = 'None', \
    color = 'red', label = 'Test')
    
    plt.show()

def f(x, S):
    return len(S[S <= x])/len(S)

def plot_cum_prob(cTrain, cTest):
    ax = plt.gca()
    
    ax.set_xlabel('$N_f$ - Observed')
    ax.set_ylabel('Cumulative Probability')
    
    ax.set_ylim(0, 1.1)
    ax.set_xlim(100, 20000)
    ax.set_xscale('log')
    
    t = np.linspace(100, 20000, 20000)
    
    t_train = [100] + cTrain.tolist() + [20000]
    t_test = [100] + cTest.tolist() + [20000]
    
    ax.step(t_train, [f(x, cTrain) for x in t_train], where = 'post', lw = 0.8, color = 'blue', label = 'Train')  
    ax.step(t_test, [f(x, cTest) for x in t_test], where = 'post', lw = 0.8, color = 'red', label = 'Test')  
    
    # gTrain = np.array([f(x, cTrain) for x in t])
    # gTest = np.array([f(x, cTest) for x in t])
    # ax.plot(t, gTrain, lw = 0.8, color = 'blue', label = 'Train')
    # ax.plot(t, gTest, lw = 0.8, color = 'red', label = 'Test')  
    # KS = max(abs(gTrain-gTest))
    # print(KS)
    
    ax.legend(framealpha = 1, edgecolor = 'None')
    
    plt.show()

Data = fd_to_df(fatigue_data.data).sort_values(by=['Temps', 'Strains'])

cycles = [get_nf(sample, from_sample=True) for sample in Data.Samples]

Data = Data.assign(Cycles = cycles)

Data_850, Data_950 = Data[Data.Temps == 850], Data[Data.Temps == 950]

test_samples = []
for st in Data_850.Strains.unique():
    tmp = Data_850[Data_850.Strains == st]
    test_samples.append(tmp.Samples.sample().iloc[0])

for st in Data_950.Strains.unique():
    tmp = Data_950[Data_950.Strains == st]
    test_samples.append(tmp.Samples.sample().iloc[0])
    
test_samples = ['4313', '41615', '4168', '41622', '41621', '41619', 'F12', '435', 'B13', 'E12', 'J6']

Data_test = Data[Data.Samples.isin(test_samples)].sort_values(by=['Cycles'])
Data_train = Data[~Data.Samples.isin(test_samples)].sort_values(by='Cycles')

cTest = Data_test.Cycles
cTrain = Data_train.Cycles

if plot:
    print('Summary Statistics:\tMean\tMedian\tStd\t\tMin\t\tMax')
    print(f'Test Data:\t\t\t{cTest.mean():.1f}\t{cTest.median():.1f}\t{cTest.std():.1f}\t{cTest.min():.1f}\t{cTest.max():.1f}')
    print(f'Train Data:\t\t\t{cTrain.mean():.1f}\t{cTrain.median():.1f}\t{cTrain.std():.1f}\t{cTrain.min():.1f}\t{cTrain.max():.1f}')
    
    plot_test_train(cTrain, cTest)
    plot_cum_prob(cTrain, cTest)

test_idx['best'] = cTest.index
train_idx['best'] = cTrain.index

#%%

Data_850 = Data_850.sort_values(by=['Cycles'])
Data_950 = Data_950.sort_values(by=['Cycles'])

test_samples = []

str_idx = (len(Data_850)//2) - (5//2)
end_idx = (len(Data_850)//2) + (5//2)
tmp = Data_850.iloc[str_idx: end_idx + 1]

for i in tmp.Samples.tolist():
    test_samples.append(i)

str_idx = (len(Data_950)//2) - (6//2)
end_idx = (len(Data_950)//2) + (6//2)
tmp = Data_950.iloc[str_idx: end_idx + 1]

for i in tmp.Samples.tolist():
    test_samples.append(i)

Data_test = Data[Data.Samples.isin(test_samples)].sort_values(by=['Cycles'])
Data_train = Data[~Data.Samples.isin(test_samples)].sort_values(by='Cycles')

cTest = Data_test.Cycles
cTrain = Data_train.Cycles
    
if plot:
    print('Summary Statistics:\tMean\tMedian\tStd\t\tMin\t\tMax')
    print(f'Test Data:\t\t\t{cTest.mean():.1f}\t{cTest.median():.1f}\t{cTest.std():.1f}\t{cTest.min():.1f}\t{cTest.max():.1f}')
    print(f'Train Data:\t\t\t{cTrain.mean():.1f}\t{cTrain.median():.1f}\t{cTrain.std():.1f}\t{cTrain.min():.1f}\t{cTrain.max():.1f}')
    
    plot_test_train(cTrain, cTest)
    plot_cum_prob(cTrain, cTest)

test_idx['out'] = cTest.index
train_idx['out'] = cTrain.index

#%%

Data_850 = Data_850.sort_values(by=['Cycles'])
Data_950 = Data_950.sort_values(by=['Cycles'])

test_samples = []

for i in range(5):
    if i%2 == 0:
        test_samples.append(Data_850.Samples.iloc[i//2])
    else:
        test_samples.append(Data_850.Samples.iloc[-(i+1)//2])

for i in range(6):
    if i%2 == 0:
        test_samples.append(Data_950.Samples.iloc[i//2])
    else:
        test_samples.append(Data_950.Samples.iloc[-(i+1)//2])

Data_test = Data[Data.Samples.isin(test_samples)].sort_values(by=['Cycles'])
Data_train = Data[~Data.Samples.isin(test_samples)].sort_values(by='Cycles')

cTest = Data_test.Cycles
cTrain = Data_train.Cycles

if plot:
    print('Summary Statistics:\tMean\tMedian\tStd\t\tMin\t\tMax')
    print(f'Test Data:\t\t\t{cTest.mean():.1f}\t{cTest.median():.1f}\t{cTest.std():.1f}\t{cTest.min():.1f}\t{cTest.max():.1f}')
    print(f'Train Data:\t\t\t{cTrain.mean():.1f}\t{cTrain.median():.1f}\t{cTrain.std():.1f}\t{cTrain.min():.1f}\t{cTrain.max():.1f}')
    
    plot_test_train(cTrain, cTest)
    plot_cum_prob(cTrain, cTest)

test_idx['in'] = cTest.index
train_idx['in'] = cTrain.index

#%%

Data_850 = Data_850.sort_values(by=['Cycles'])
Data_950 = Data_950.sort_values(by=['Cycles'])

test_samples = []

for i in range(5):
    test_samples.append(Data_850.Samples.iloc[i])

for i in range(6):
    test_samples.append(Data_950.Samples.iloc[i])

Data_test = Data[Data.Samples.isin(test_samples)].sort_values(by=['Cycles'])
Data_train = Data[~Data.Samples.isin(test_samples)].sort_values(by='Cycles')

cTest = Data_test.Cycles
cTrain = Data_train.Cycles
    
if plot:
    print('Summary Statistics:\tMean\tMedian\tStd\t\tMin\t\tMax')
    print(f'Test Data:\t\t\t{cTest.mean():.1f}\t{cTest.median():.1f}\t{cTest.std():.1f}\t{cTest.min():.1f}\t{cTest.max():.1f}')
    print(f'Train Data:\t\t\t{cTrain.mean():.1f}\t{cTrain.median():.1f}\t{cTrain.std():.1f}\t{cTrain.min():.1f}\t{cTrain.max():.1f}')
    
    plot_test_train(cTrain, cTest)
    plot_cum_prob(cTrain, cTest)

test_idx['high'] = cTest.index
train_idx['high'] = cTrain.index

#%%

Data_850 = Data_850.sort_values(by=['Cycles'])
Data_950 = Data_950.sort_values(by=['Cycles'])

test_samples = []

for i in range(5):
    test_samples.append(Data_850.Samples.iloc[-i-1])

for i in range(6):
    test_samples.append(Data_950.Samples.iloc[-i-1])

Data_test = Data[Data.Samples.isin(test_samples)].sort_values(by=['Cycles'])
Data_train = Data[~Data.Samples.isin(test_samples)].sort_values(by='Cycles')

cTest = Data_test.Cycles
cTrain = Data_train.Cycles

if plot:
    print('Summary Statistics:\tMean\tMedian\tStd\t\tMin\t\tMax')
    print(f'Test Data:\t\t\t{cTest.mean():.1f}\t{cTest.median():.1f}\t{cTest.std():.1f}\t{cTest.min():.1f}\t{cTest.max():.1f}')
    print(f'Train Data:\t\t\t{cTrain.mean():.1f}\t{cTrain.median():.1f}\t{cTrain.std():.1f}\t{cTrain.min():.1f}\t{cTrain.max():.1f}')
    
    plot_test_train(cTrain, cTest)
    plot_cum_prob(cTrain, cTest)

test_idx['low'] = cTest.index
train_idx['low'] = cTrain.index