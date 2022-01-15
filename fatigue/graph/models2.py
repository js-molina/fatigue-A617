import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from .helper import chi_ratio
from ..models import TEMPS
from ..finder import fatigue_data
from ..models2 import get_nf

def graph_model(model):
        
    tdata, f, lc = model._get_plot_params()
    model_type = model._get_model_type()
    
    ax = plt.gca()
    
    if model_type == 'morrow':
        ax.set_ylabel(r"Cycles to Failure")
        ax.set_xlabel(r"Plastic Strain Energy Density (MJ/kg$^3$)")
        ax.set_ylim(100, 20000)
        ax.set_xlim(0.1, 20)
        
    if model_type == 'pl_manson':
        ax.set_ylabel(r"Cycles to Failure")
        ax.set_xlabel(r"Plastic Strain (\%)")
        ax.set_ylim(100, 20000)
        ax.set_xlim(0.01, 10)
        
    if model_type == 'c_manson':
        ax.set_ylabel(r"Cycles to Failure")
        ax.set_xlabel(r"Elastic Strain (\%)")
        ax.set_ylim(100, 20000)
        ax.set_xlim(0.06, 0.5)

    colors = ['b', 'r']
    markers = ['x', 'o']
    
    for i, (x, y) in enumerate(tdata):
        xfine = np.linspace(min(x), max(x))
        ax.plot(xfine, f[i](xfine), color = colors[i], lw = 0.8)
        ax.loglog(x, y, marker = markers[i], markersize = 5, ls = 'None', \
                  markeredgecolor = colors[i], markerfacecolor = 'None', markeredgewidth = 1)
        
    for i, (A, a) in enumerate(lc):
        ax.loglog([-10], [-1], marker = markers[i], color = colors[i], markersize = 5,\
                  markerfacecolor = 'None', markeredgewidth = 1,\
            lw = 0.7, label = '\SI{%d}{\celsius}\quad $y = %dx^{%.2f}$'%(TEMPS[i], A, a))

    ax.legend(framealpha = 1, edgecolor = 'None')
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    plt.show()
    
def graph_prediction(model):
        
    ax = plt.gca()
    
    model_type = model._get_model_type()
    
    if model_type == 'morrow':
        ax.set_title('Morrow Model')
    elif model_type == 'pl_manson':
        ax.set_title('Inelastic Coffin Manson Model')
    elif model_type == 'c_manson':
        ax.set_title('Elastic Coffin Manson Model')
    
    ax.set_aspect('equal')
    
    ax.set_ylim(100, 20000)
    ax.set_xlim(100, 20000)

    x_obs = [x[1] for x in model.values]
    y = [x[0] for x in model.values]
    x_pred = [model.pred[i](y_) for i, y_ in enumerate(y)]
    
    colors = ['b', 'r']
    markers = ['x', 'o']
    labels = []
    
    for i in range(len(TEMPS)):
        s1 = '\SI{%d}{\celsius}'%TEMPS[i]
        s2 = '%.3f'%chi_ratio(x_pred[i], x_obs[i])
        labels.append(' -- $\chi^2 =\ $'.join([s1, s2]))
    
    ax = plt.gca()
    
    ax.set_xlabel('Predicted $N_f$')
    ax.set_ylabel('Measured $N_f$')
    
    for i in range(2):
        ax.loglog(x_pred[i], x_obs[i], marker = markers[i], markersize = 5, ls = 'None', \
        markeredgecolor = colors[i], markerfacecolor = 'None', markeredgewidth = 1, label = labels[i])
    
    ax.plot([100, 20000], [100, 20000], lw = 2, color = 'k')
    ax.fill_between([100, 20000], 100, [100, 20000], color = 'k', alpha = 0.1)

    ax.legend(framealpha = 1, edgecolor = 'None')
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    plt.show()
    
def graph_nn_prediction(data, log = False):
    d = np.load(data)
    y_obs, y_pred = d['y_obs'], d['y_pred']
    
    c8 = []
    c9 = []
    
    for test in fatigue_data.data:
        if test.Temp == 850:
            c8.append(get_nf(test))
        else:
            c9.append(get_nf(test))
    
    t8 = np.setdiff1d(np.rint(y_obs).astype('int') , np.array(c9))
    t9 = np.setdiff1d(np.rint(y_obs).astype('int') , np.array(c8))
    
    i8 = np.array([np.where(np.rint(y_obs) == t) for t in t8]).flatten()
    i9 = np.array([np.where(np.rint(y_obs) == t) for t in t9]).flatten()
    
    colors = ['b', 'r']
    markers = ['x', 'o']
    labels = []
    
    x_pred = [y_pred[i8], y_pred[i9]]
    x_obs = [y_obs[i8], y_obs[i9]]
    
    for i in range(len(TEMPS)):
        s1 = '\SI{%d}{\celsius}'%TEMPS[i]
        s2 = '%.3f'%chi_ratio(x_pred[i], x_obs[i])
        labels.append(' -- $\chi^2 =\ $'.join([s1, s2]))
    
    ax = plt.gca()
    
    ax.set_xlabel('Predicted $N_f$')
    ax.set_ylabel('Measured $N_f$')
    
    ax.set_ylim(100, 12000)
    ax.set_xlim(100, 12000)
    
    if log:
        ax.set_ylim(100, 20000)
        ax.set_xlim(100, 20000)
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    ax.set_aspect('equal')
    
    for i in range(2):
        ax.plot(x_pred[i], x_obs[i], marker = markers[i], markersize = 5, ls = 'None', \
        markeredgecolor = colors[i], markerfacecolor = 'None', markeredgewidth = 1, label = labels[i])
    
    ax.plot([100, 20000], [100, 20000], lw = 2, color = 'k')
    ax.fill_between([100, 20000], 100, [100, 20000], color = 'k', alpha = 0.1)
    
    ax.legend(framealpha = 1, edgecolor = 'None')
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    plt.show()
    
def graph_nn_pred_strain(data, log = False):
    d = np.load(data)
    y_obs, y_pred = d['y_obs'], d['y_pred']
    
    strain_data = {}
    
    for test in fatigue_data.data:
        c = get_nf(test)
        for j, el in enumerate(np.rint(y_obs).astype('int')):
            if c == el:
                strain_data.setdefault((test.Temp, test.Strain), [])
                strain_data[(test.Temp, test.Strain)].append((el, y_pred[j]))
                break
    
    ax = plt.gca()
    
    ax.set_xlabel('Predicted $N_f$')
    ax.set_ylabel('Measured $N_f$')
        
    ax.set_ylim(100, 12000)
    ax.set_xlim(100, 12000)
    
    if log:
        ax.set_ylim(100, 20000)
        ax.set_xlim(100, 20000)
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    ax.set_aspect('equal')
    
    dict_marker = {0.3 : 'o', 0.4: '^', 0.6: 'v', 1 : 's', 2: 'x', 3: '+'}
    dict_color = {850 : 'blue', 950: 'red'}
    
    for key, value in strain_data.items():
        color, marker = dict_color[key[0]], dict_marker[key[1]]
        x, y = zip(*value)
        ax.plot(x, y, marker = marker, markersize = 6, ls = 'None', \
        markeredgecolor = color, markerfacecolor = 'None', markeredgewidth = 1.5, label = '%d - %.1f'%(key[0], key[1]))
    
    ax.plot([100, 20000], [100, 20000], lw = 2, color = 'k')
    ax.fill_between([100, 20000], 100, [100, 20000], color = 'k', alpha = 0.1)
    
    # ax.legend(framealpha = 1, edgecolor = 'None')
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    plt.show()
    
def graph_nn_pred_all(data, log = False):
    d = np.load(data)
    y_obs, y_pred = d['y_obs'], d['y_pred']
    
    strain_data = {}
    
    for test in fatigue_data.data:
        c = get_nf(test)
        for j, el in enumerate(np.rint(y_obs).astype('int')):
            if c == el:
                strain_data.setdefault((test.Temp, test.Strain, test.Rate), [])
                strain_data[(test.Temp, test.Strain, test.Rate)].append((y_pred[j], el))
                break
    
    fig, ax = plt.subplots(figsize=(8,4))
    fig.subplots_adjust(right=0.5)
    
    ax.set_xlabel('Predicted $N_f$')
    ax.set_ylabel('Measured $N_f$')
        
    ax.set_ylim(100, 12000)
    ax.set_xlim(100, 12000)
    
    if log:
        ax.set_ylim(100, 20000)
        ax.set_xlim(100, 20000)
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    # ax.set_aspect('equal')
    ax.fill_between([100, 20000], 100, [100, 20000], color = 'k', alpha = 0.1)
    ax.plot([100, 20000], [100, 20000], lw = 2, color = 'k')
    
    colors = plt.cm.gist_rainbow(np.linspace(0,1,6))
    strain_vals = [0.3, 0.4, 0.6, 1, 2, 3]
    
    dict_color = dict(zip(strain_vals, colors))
    dict_marker = {0.001 : 'o', 0.0001: '^', 1e-5: 's'}
    dict_shade = {850 : False, 950: True}
    
    for key, value in strain_data.items():
        temp, strain, rate = key
        color, marker, shade = dict_color[strain], dict_marker[rate], dict_shade[temp]
        
        if shade:
            facecol = color
        else:
            facecol = 'None'
        
        x, y = zip(*value)
        
        ax.plot(x, y, marker = marker, markersize = 6, ls = 'None', \
        markeredgecolor = color, markerfacecolor = facecol, markeredgewidth = 1.5)
            
    strain_elements = [Patch(facecolor= val, edgecolor=val, label='{:.1f}\%'.format(key))
                       for key, val in dict_color.items()]
    
    rate_elements = [Line2D([0], [0], marker=val, color='k', label='{:.5f}'.format(key), ls = 'None', \
                     markerfacecolor='None', markeredgewidth = 1.5, markersize=6) for key, val in dict_marker.items()]
        
    temp_elements = []
    
    for key, val in dict_shade.items():
        cl = 'None'
        if val:
            cl = 'k'
        temp_elements.append(Line2D([0], [0], marker='h', color='k', label='\SI{%d}{\celsius}'%key, ls = 'None', \
             markerfacecolor=cl, markeredgewidth = 1.5, markersize=6))
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    l1 = ax.legend(title = 'Strain Range', 
              handles=strain_elements,
              loc='center right',
              bbox_to_anchor=(1.4, 0.525),
              edgecolor = 'None')
    
    l2 = ax.legend(title = 'Strain Rate (Hz)', 
          handles=rate_elements,
          loc='center right',
          bbox_to_anchor=(1.45, 0.1),
          edgecolor = 'None')
    
    ax.legend(title = 'Temperature', 
          handles=temp_elements,
          loc='center right',
          bbox_to_anchor=(1.4, 0.9),
          edgecolor = 'None')
    
    ax.add_artist(l1)
    ax.add_artist(l2)
    
    # path = r'D:\WSL\ansto\figs'
    
    # plt.savefig(os.path.join(path, 'dynamic.pdf'))
    plt.show()