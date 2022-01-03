import matplotlib.pyplot as plt
import numpy as np

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
    
def graph_nn_prediction(data):
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
    
    ax.set_ylim(100, 20000)
    ax.set_xlim(100, 20000)
    
    ax.set_aspect('equal')
    
    for i in range(2):
        ax.loglog(x_pred[i], x_obs[i], marker = markers[i], markersize = 5, ls = 'None', \
        markeredgecolor = colors[i], markerfacecolor = 'None', markeredgewidth = 1, label = labels[i])
    
    ax.plot([100, 20000], [100, 20000], lw = 2, color = 'k')
    ax.fill_between([100, 20000], 100, [100, 20000], color = 'k', alpha = 0.1)
    
    ax.legend(framealpha = 1, edgecolor = 'None')
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    plt.show()