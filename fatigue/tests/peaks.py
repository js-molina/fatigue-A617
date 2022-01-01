import matplotlib.pyplot as plt
import numpy as np
from ..finder import fatigue_data
from ..graph import get_cycles_from_test, graph_cycle
from ..strain import scuffed_plastic_energy_pts, get_plastic_energy
from ..networks import *

def test_strain_from_peaks(test, l_cycle):
    x = stress(test)
    
    if test.Temp == 850:
        E = 153e3
    else:
        E = 144e3 
    
    x = app_elastic_e(x, E)
    x = app_plastic_e(x, test.Strain)

    for el in l_cycle:
        dt = x[x['cycle'] == el]
        print('Cycle = %d'%el)
        print('Elastic Strain = %.2f'%(dt.elastic))
        print('Plastic Strain = %.2f'%(dt.plastic))
    
    
def test_scuffed_energy(trial, l_cycle):
    
    df = test_some_peak_data(trial)
    
    cycles = get_cycles_from_test(trial)

    for cycle, el in zip(cycles, l_cycle):
        
        dt = df[df['cycle'] == el]
        
        x, y = scuffed_plastic_energy_pts(dt.max_s.iloc[0], dt.min_s.iloc[0], dt.plastic.iloc[0], trial.Strain)
        
        ax = plt.gca()
        ax.set_title(cycle['Cycle Label'].iloc[0])
        graph_cycle(cycle, 'gray', ax)

        ax.plot(x/100, y, 'b*')
        ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
        plt.show()
        
        es0 = get_plastic_energy(cycle)
        
        print('%.3f'%es0)
        
        es1 = (x[-1]-x[-2])*(y[1]-y[-1])/100
        es2 = (x[-1]-x[-2])*np.mean(np.abs([y[1], y[-1]]))/100
        
        print('%.3f\t%.2f'%(es1, abs(es1-es0)/es0*100))
        print('%.3f\t%.2f'%(es2, abs(es2-es0)/es0*100))
        
        es3 = (es1+es2)/2
        
        print('%.3f\t%.2f'%(es3, abs(es3-es0)/es0*100))


def test_some_peak_data(test):
    x = stress(test)
    
    if test.Temp == 850:
        E = 153e3
    else:
        E = 144e3 
    
    x = app_elastic_e(x, E)
    x = app_plastic_e(x, test.Strain)
    
    x = app_bavg(x)
    x = app_favg(x)
    
    return x

def test_features(data):
    x = []
    for test in data:
        x.append(features(test).reset_index(drop = True))
    return x
    
def test_some_data(test):
    
    lw_ = 1
    
    x = test_some_peak_data(test)
    
    l = x.cycle.iloc[-1]
    
    print(test.Sample, 'l = %d'%l, sep = '\t')
    
    ax = plt.gca()
    
    ax.set_xlim([10, l*1.1])
    
    ax.plot(x.cycle, x.min_s, 'r-', lw = lw_, alpha = 0.3)
    ax.plot(x.cycle, x.max_s, 'b-', lw = lw_, alpha = 0.3)
    
    # ax.plot(x.cycle, x.min_s_b100, 'r--', lw = lw_)
    # ax.plot(x.cycle, x.max_s_b100, 'b--', lw = lw_)
    
    # ax.plot(x.cycle, x.min_s_b50, 'r-.', lw = lw_)
    # ax.plot(x.cycle, x.max_s_b50, 'b-.', lw = lw_)
    
    # ax.plot(x.cycle, x.min_s_b10, 'r:', lw = lw_)
    # ax.plot(x.cycle, x.max_s_b10, 'b:', lw = lw_)
    
    # ax.plot(x.cycle, x.min_s_b30p, 'r--', lw = lw_)
    # ax.plot(x.cycle, x.max_s_b30p, 'b--', lw = lw_)
    
    # ax.plot(x.cycle, x.min_s_b20p, 'r-.', lw = lw_)
    # ax.plot(x.cycle, x.max_s_b20p, 'b-.', lw = lw_)
    
    # ax.plot(x.cycle, x.min_s_f10p, 'r:', lw = lw_)
    # ax.plot(x.cycle, x.max_s_f10p, 'b:', lw = lw_)
    
    ax.plot(x.cycle, x.min_s_f10, 'r:', lw = lw_)
    ax.plot(x.cycle, x.max_s_f10, 'b:', lw = lw_)
    
    # ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    plt.show()    