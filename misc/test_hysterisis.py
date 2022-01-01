import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

E = 201.3e3; H = 1000; n = 0.112

def eApos(sa, ee):
    return sa/E + (sa/H)**(1/n) - ee
def eAneg(sa, ee):
    return sa/E - (-sa/H)**(1/n) - ee
def deA(dsa, dee):
    return dsa/E + 2*(dsa/(2*H))**(1/n) - dee
def f(s, e_min, s_min):
    return e_min + 2*((s-s_min)/(2*E) +((s-s_min)/(2*H))**(1/n))
def g(s, e_max, s_max):
    return e_max - 2*((s_max-s)/(2*E) +((s_max-s)/(2*H))**(1/n))

S = []
dT = []
T = [-0.0248, 0.0067, -0.0184, -0.0056, -0.0136, 0.0232, 0.0014, 0.0167, -0.0248]
T_or = [-0.0248, 0.0067, -0.0184, -0.0056, -0.0248, 0.0232, 0.0014, 0.0167, -0.0248]

S.append(fsolve(lambda x: eAneg(x, T[0]), x0 = -4000)[0])

dS = []

for i, e in enumerate(T[1:]):
    if i%2 == 0:
        r = 1
    else:
        r = -1
        
    de = r*(e-T_or[i])
    ds = fsolve(lambda x: deA(x, de), x0 = 2000)[0]
    s = S[i]+r*ds
    
    dT.append(de)
    dS.append(ds)
    S.append(s)
    
i = 1

ax = plt.gca()

s = np.linspace(0, S[0])
e = eAneg(s, 0)
ax.plot(e, s, lw = 0.7, color = 'red')

while i < len(S):
    s = np.linspace(S[i-1], S[i])
    if i%2 == 1:
        e = f(s, e[-1], s[0])
        ax.plot(e, s, lw = 0.7, color = 'blue')
    else:
        e = g(s, e[-1], s[0])
        ax.plot(e, s, lw = 0.7, color = 'blue')
    i += 1

ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)

plt.show()

