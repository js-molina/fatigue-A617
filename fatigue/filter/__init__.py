import matplotlib.pyplot as plt
import pandas as pd
from ..graph import get_cycles_from_test, graph_cycle

import numpy as np
from scipy.optimize import curve_fit
from tsmoothie.smoother import ConvolutionSmoother

def test_filter(test, lowest_cov = False):
    for cycle in get_cycles_from_test(test):
    
        cycle = cycle.copy().reset_index()
        
        i2 = cycle.loc[cycle.Strain == max(cycle.Strain)].index[0]
        i3 = cycle.loc[cycle.Strain == min(cycle.Strain)].index[0]
        
        if i2 > 20 and i2 < len(cycle) - 20:
            t1 = cycle.loc[i3:].copy()
            t2 = cycle.loc[:i2].copy()
            t3 = cycle.loc[i2:i3].copy()
        else:
            t1 = cycle.loc[:i3].copy()
            t2 = []
            t3 = cycle.loc[i3:].copy()
        
        t1['Elapsed Time Sec'] -= t1['Elapsed Time Sec'].iloc[0]
        t3['Elapsed Time Sec'] -= t3['Elapsed Time Sec'].iloc[0]
        
        if len(t2) > 0:
            t2['Elapsed Time Sec'] -= t2['Elapsed Time Sec'].iloc[0] - t1['Elapsed Time Sec'].iloc[-1]
            ten = pd.concat([t1, t2], ignore_index=True)
            com = t3.reset_index()
        else:
            ten = t3.reset_index()
            com = t1.reset_index()
            
        # ax = graph_cycle(t1, 'blue')
        # if len(t2) > 0:
        #     ax = graph_cycle(t2, 'red', ax)
        # graph_cycle(t3, 'green', ax, flush = True)
        
        win_len = 25
        
        sm_ten = ConvolutionSmoother(window_len=win_len, window_type='ones')
        sm_ten.smooth(ten['Stress Mpa'])
        
        sm_com = ConvolutionSmoother(window_len=win_len, window_type='ones')
        sm_com.smooth(com['Stress Mpa'])
        
        if test.Temp == 850:
            E = 153e3
        else:
            E = 144e3
    
        e_m_ten, s_m_ten = map(min, [ten.Strain, ten['Stress Mpa']])
        e_M_ten, s_M_ten = map(max, [ten.Strain, ten['Stress Mpa']])
        
        e_m_com, s_m_com = map(min, [com.Strain, com['Stress Mpa']])
        e_M_com, s_M_com = map(max, [com.Strain, com['Stress Mpa']])
    
        def ften(x, H, n):
            return e_m_ten +  2*((x-s_m_ten)/(2*E) +((x-s_m_ten)/(2*H))**(1/n))
        def gcom(x, H, n):
            return e_M_com - 2*((s_M_com-x)/(2*E) +((s_M_com-x)/(2*H))**(1/n))
    
    
        s_ten = pd.DataFrame(zip(sm_ten.smooth_data[0], ten.Strain), columns = ['stress', 'strain'])
        s_com = pd.DataFrame(zip(sm_com.smooth_data[0], com.Strain), columns = ['stress', 'strain'])
        
        popt_s_ten, pcov_s_ten = curve_fit(ften, s_ten.stress, s_ten.strain)
        popt_s_com, pcov_s_com = curve_fit(gcom, s_com.stress, s_com.strain)
        
        popt_ten, pcov_ten = curve_fit(ften, ten['Stress Mpa'], ten.Strain)
        popt_com, pcov_com = curve_fit(gcom, com['Stress Mpa'], com.Strain)
        
    
        def eApos(sa, ee, H, n):
            return sa/E + (sa/H)**(1/n) - ee
        def eAneg(sa, ee, H, n):
            return sa/E - (-sa/H)**(1/n) - ee
        def deA(dsa, dee, H, n):
            return dsa/E + 2*(dsa/(2*H))**(1/n) - dee
        def f(e_min, s_min, s, H, n):
            return e_min + 2*((s-s_min)/(2*E) +((s-s_min)/(2*H))**(1/n))
        def g(e_max, s_max, s, H, n):
            return e_max - 2*((s_max-s)/(2*E) +((s_max-s)/(2*H))**(1/n))
    
    
    
        if np.mean(pcov_s_ten) <= np.mean(pcov_ten):
            popt_ten = popt_s_ten
        if np.mean(pcov_s_com) <= np.mean(pcov_com):
            popt_com = popt_s_com
            
        if np.mean(pcov_ten) >= np.mean(pcov_com):
            popt = popt_com
        else:
            popt = popt_ten
        
        if lowest_cov:
            popt_com = popt
            popt_ten = popt
        
        ax = plt.gca()
        
        ax.plot(ten.Strain, ten['Stress Mpa'], color='blue', lw = 0.5, alpha = 0.5)
        ax.plot(com.Strain, com['Stress Mpa'], color='red', lw = 0.5, alpha = 0.5)
        
        e_m, s_m = map(min, [ten.Strain, ten['Stress Mpa']])
        e_M, s_M = map(max, [ten.Strain, ten['Stress Mpa']])
        
        s_ten = np.linspace(s_m, s_M, 20000)
        
        e_ten = f(e_m, s_m, s_ten, *popt_ten)
        
        ax.plot(e_ten[e_ten < e_M], s_ten[e_ten < e_M], lw = 2, color = 'blue')
        
        e_m, s_m = map(min, [com.Strain, com['Stress Mpa']])
        e_M, s_M = map(max, [com.Strain, com['Stress Mpa']])
        
        s_com = np.linspace(s_M, s_m, 20000)
        
        e_com = g(e_M, s_M, s_com, *popt_com)
        ax.plot(e_com[e_com > e_m], s_com[e_com > e_m], lw = 2, color = 'red')
        
        ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
        
        plt.show()

