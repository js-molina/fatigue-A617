import matplotlib.pyplot as plt
import pandas as pd
from fatigue.finder import fatigue_data
import fatigue.graph as gr

import numpy as np
from scipy.optimize import curve_fit
from tsmoothie.smoother import ConvolutionSmoother
from functools import partial

test = fatigue_data.get_data(950)[-5]

cycle = gr.get_cycles_from_test(test)[1].reset_index()

i2 = cycle.loc[cycle.Strain == max(cycle.Strain)].index[0]
i3 = cycle.loc[cycle.Strain == min(cycle.Strain)].index[0]

print(i2, i3)

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
    

a = cycle.loc[i2]; b = cycle.loc[i3]

a = [a.Strain, a['Stress Mpa']]
b = [b.Strain, b['Stress Mpa']]

ax = gr.graph_cycle(ten, 'blue')
ax = gr.graph_cycle(com, 'red')

plt.show()

#%%

ax = plt.gca()

win_len = 25

sm_ten = ConvolutionSmoother(window_len=win_len, window_type='ones')
sm_ten.smooth(ten['Stress Mpa'])

ax.plot(ten.Strain, sm_ten.data[0], color='gray', lw = 0.7)
ax.plot(ten.Strain, sm_ten.smooth_data[0],color='blue', lw = 1)

sm_com = ConvolutionSmoother(window_len=win_len, window_type='ones')
sm_com.smooth(com['Stress Mpa'])

ax.plot(com.Strain, sm_com.data[0], color='gray', lw = 0.7)
ax.plot(com.Strain, sm_com.smooth_data[0],color='red', lw = 1)

plt.show()


#%%

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


# m1 = s_ten.stress
# m2 = s_com.stress

m1 = ten['Stress Mpa']
m2 = com['Stress Mpa']


ax = plt.gca()

ax.plot(m1, s_ten.strain, color = 'xkcd:orange', ls = '-', lw = 0.7)
ax.plot(m1, ften(m1, *popt_ten), 'g--', lw = 0.7)
ax.plot(m1, ften(m1, *popt_s_ten), '--', lw = 0.7)

plt.show()

ax = plt.gca()

ax.plot(m2, s_com.strain, color = 'xkcd:orange', ls = '-', lw = 0.7)
ax.plot(m2, gcom(m2, *popt_com), 'g--', lw = 0.7)
ax.plot(m2, gcom(m2, *popt_s_com), '--', lw = 0.7)

plt.show()


#%%

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
# else:
#     popt = popt_ten

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

