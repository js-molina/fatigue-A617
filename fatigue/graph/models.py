import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from .helper import chi_ratio
from ..models import TEMPS

def graph_model(model, save = None):
        
    tdata0, tdata1, f, lc = model._get_plot_params()
    model_type = model._get_model_type()
    
    fig, ax = plt.subplots(figsize=(4,4))
    
    if model_type == 'morrow':
        ax.set_xlabel(r"Cycles to Failure")
        ax.set_ylabel(r"Plastic Strain Energy Density (MJ/kg$^3$)")
        ax.set_xlim(100, 20000)
        ax.set_ylim(0.1, 20)
        
    if model_type == 'pl_manson':
        ax.set_xlabel(r"Cycles to Failure")
        ax.set_ylabel(r"Plastic Strain (\%)")
        ax.set_xlim(100, 20000)
        ax.set_ylim(0.01, 10)
        
    if model_type == 'c_manson':
        ax.set_xlabel(r"Cycles to Failure")
        ax.set_ylabel(r"Elastic Strain (\%)")
        ax.set_xlim(100, 20000)
        ax.set_ylim(0.06, 0.5)

    colors = ['b', 'r']
    markers = ['x', 'o']
    
    for i, (x, y) in enumerate(tdata0):
        xfine = np.linspace(min(x), max(x))
        ax.plot(xfine, f[i](xfine), color = colors[i], lw = 0.8)
        ax.loglog(x, y, marker = 'x', markersize = 6, ls = 'None', \
                  markeredgecolor = colors[i], markerfacecolor = 'None', markeredgewidth = 1)
    
    for i, (x, y) in enumerate(tdata1):
        xfine = np.linspace(min(x), max(x))
        ax.plot(xfine, f[i](xfine), color = colors[i], lw = 0.8)
        ax.loglog(x, y, marker = 'o', markersize = 6, ls = 'None', \
                  markeredgecolor = colors[i], markerfacecolor = 'None', markeredgewidth = 1)
    
    for i, (A, a) in enumerate(lc):
        ax.loglog([-10], [-1], color = colors[i], lw = 1.7, \
                  label = '\SI{%d}{\celsius}\quad $y = %dx^{%.2f}$'%(TEMPS[i], A, a))

    dict_marker = {'Train' : 'x', 'Test': 'o'}
    elements = [Line2D([0], [0], marker=val, color='k', label='{}'.format(key), ls = 'None', \
                     markerfacecolor='None', markeredgewidth = 1.5, markersize=5) for key, val in dict_marker.items()]


    l1 = ax.legend(handles=elements, framealpha = 1, ncol = 2,
              bbox_to_anchor=(0.77, 1.12), edgecolor = 'None', fontsize = 10)

    ax.legend(framealpha = 1, edgecolor = 'None', loc = 0)
    
    ax.add_artist(l1)
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    if save:
        path = r'D:\WSL\ansto\figs'
        plt.savefig(os.path.join(path, save), bbox_inches = 'tight')
    
    plt.show()
    
def graph_prediction(model, save = None):
        
    ax = plt.gca()
    
    model_type = model._get_model_type()
    
    # if model_type == 'morrow':
    #     ax.set_title('Morrow Model')
    # elif model_type == 'pl_manson':
    #     ax.set_title('Inelastic Coffin Manson Model')
    # elif model_type == 'c_manson':
    #     ax.set_title('Elastic Coffin Manson Model')
    
    ax.set_aspect('equal')
    
    ax.set_ylim(100, 20000)
    ax.set_xlim(100, 20000)
    
    x_obs0 = [x[0] for x in model.values0]
    y0 = [x[1] for x in model.values0]
    x_pred0 = [model.pred[i](y_) for i, y_ in enumerate(y0)]
    
    x_obs1 = [x[0] for x in model.values1]
    y1 = [x[1] for x in model.values1]
    x_pred1 = [model.pred[i](y_) for i, y_ in enumerate(y1)]
    
    colors = ['b', 'r']
    markers = ['x', 'o']
    labels = []
    
    x_pred = np.concatenate((x_pred0, x_pred1), dtype = 'object')
    x_obs = np.concatenate((x_obs0, x_obs1), dtype = 'object')
    
    for i in range(len(TEMPS)):
        s1 = '\SI{%d}{\celsius}'%TEMPS[i]
        s2 = '%.3f'%chi_ratio(x_pred[i], x_obs[i])
        labels.append(' -- $\chi^2 =\ $'.join([s1, s2]))
    
    ax = plt.gca()
    
    ax.set_xlabel('Predicted $N_f$')
    ax.set_ylabel('Measured $N_f$')
    
    for i in range(2):
        ax.loglog(x_pred0[i], x_obs0[i], marker = 'x', markersize = 6, ls = 'None', \
        markeredgecolor = colors[i], markerfacecolor = 'None', markeredgewidth = 1, label = labels[i])
            
    for i in range(2):
        ax.loglog(x_pred1[i], x_obs1[i], marker = 'o', markersize = 6, ls = 'None', \
        markeredgecolor = colors[i], markerfacecolor = 'None', markeredgewidth = 1)
    
    ax.plot([100, 20000], [100, 20000], lw = 2, color = 'k')
    ax.fill_between([100, 20000], 100, [100, 20000], color = 'k', alpha = 0.1)
    
    ax.legend(framealpha = 1, edgecolor = 'None')
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    y_pred = np.concatenate((x_pred0[0], x_pred0[1], x_pred1[0], x_pred1[1]), dtype = 'object')
    y_obs = np.concatenate((x_obs0[0], x_obs0[1], x_obs1[0], x_obs1[1]), dtype = 'object')
    
    ax.set_title('$\chi^2 = %.3f$'%chi_ratio(y_pred, y_obs))
    
    if save:
        path = r'D:\WSL\ansto\figs'
        plt.savefig(os.path.join(path, save))
    
    plt.show()