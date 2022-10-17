import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import os

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import matplotlib.path as mpath
import matplotlib.gridspec as gridspec

from .helper import chi_ratio, meape
from ..models import TEMPS
from ..finder import fatigue_data
from ..models2 import get_nf

path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'
# path = r'D:\INDEX\Notes\Semester_15\MMAN4952\Thesis B\figs'
path = r'D:\INDEX\Notes\Semester_16\MMAN4953\Thesis C\img'

def graph_model(model, save_path = None):
        
    tdata, f, lc = model._get_plot_params()
    model_type = model._get_model_type()
    
    fig, ax = plt.subplots(figsize=(4,4))
    
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
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
def graph_prediction(model, save_path = None):
        
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

    x_obs = [x[1] for x in model.values]
    y = [x[0] for x in model.values]
    x_pred = [model.pred[i](y_) for i, y_ in enumerate(y)]
    
    colors = ['b', 'r']
    markers = ['x', 'o']
    labels = []
    
    for i in range(len(TEMPS)):
        s1 = '\SI{%d}{\celsius}'%TEMPS[i]
        # s2 = '%.3f'%chi_ratio(x_pred[i], x_obs[i])
        # labels.append(' -- $\chi^2 =\ $'.join([s1, s2]))
        
        s2 = '%.2f'%((abs(x_obs[i]-x_pred[i])/x_obs[i]).mean()*100) + '\%'
        labels.append(' -- $\mathbb{E} =\ $'.join([s1, s2]))
        
    
    ax = plt.gca()
    
    ax.set_xlabel('Predicted $N_f$')
    ax.set_ylabel('Measured $N_f$')
    
    for i in range(2):
        ax.loglog(x_pred[i], x_obs[i], marker = markers[i], markersize = 6, ls = 'None', \
        markeredgecolor = colors[i], markerfacecolor = 'None', markeredgewidth = 1, label = labels[i])
    
    ax.plot([100, 20000], [100, 20000], lw = 2, color = 'k')
    
    ax.plot([100, 1e5], [200, 2e5], lw = 1, ls = '--', color = 'gray')
    ax.plot([200, 2e5], [100, 1e5], lw = 1, ls = '--', color = 'gray')
    
    ax.fill_between([100, 20000], 100, [100, 20000], color = 'k', alpha = 0.1)

    ax.legend(framealpha = 1, edgecolor = 'k')
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    x_obs = np.concatenate(x_obs)
    x_pred = np.concatenate(x_pred)
    
    print((abs(x_obs-x_pred)/x_obs).mean()*100)
    
    
def graph_nn_prediction(data, log = False, v2 = False, load = True):
    if load:
        d = np.load(data)
    else:
        d = data
    if v2:
        y_obs, y_pred = d['y_obs_test'], d['y_pred_test']
    else:
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
    
    ax.plot([100, 20000], [100, 20000], lw = 2, color = 'k')
    ax.fill_between([100, 20000], 100, [100, 20000], color = 'k', alpha = 0.1)
    
    for i in range(2):
        ax.plot(x_pred[i], x_obs[i], marker = markers[i], markersize = 7, ls = 'None', \
        markeredgecolor = colors[i], markerfacecolor = 'None', markeredgewidth = 1, label = labels[i])
    
    ax.legend(framealpha = 1, edgecolor = 'None')
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    plt.show()

def graph_nn_pred_strain(data, log = False, load = True):
    if load:
        d = np.load(data)
    else:
        d = data
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
    
def graph_nn_pred_all(data, log = False, v2 = True, load = True, save = ''):
    if load:
        print(data)
        d = np.load(data)
    else:
        d = data
    if v2:
        y_obs, y_pred = d['y_obs_test'], d['y_pred_test']
    else:
        y_obs, y_pred = d['y_obs'], d['y_pred']
    
    strain_data = {}
    
    for test in fatigue_data.data:
        c = get_nf(test)
        for j, el in enumerate(np.rint(y_obs).astype('int')):
            if c == el:
                strain_data.setdefault((test.Temp, test.Strain, test.Rate), [])
                strain_data[(test.Temp, test.Strain, test.Rate)].append((y_pred[j], el))
                # break
    
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
    
    if not log:
        ax.plot([0, 6000], [0, 12000], lw = 1, ls = '--', color = 'gray')
        ax.plot([0, 12000], [0, 6000], lw = 1, ls = '--', color = 'gray')
    else:
        ax.plot([100, 20000], [200, 40000], lw = 1, ls = '--', color = 'gray')
        ax.plot([200, 40000], [100, 20000], lw = 1, ls = '--', color = 'gray')
    
    msize = 7
    
    colors = plt.cm.gist_rainbow(np.linspace(0,1,6)).tolist()
    colors[1] = 'xkcd:orange'
    colors[2] = 'xkcd:green'
    colors[3] = 'xkcd:sky blue'
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
        
        ax.plot(x, y, marker = marker, markersize = msize, ls = 'None', \
        markeredgecolor = color, markerfacecolor = facecol, markeredgewidth = 1.5)
            
    strain_elements = [Patch(facecolor= val, edgecolor=val, label='{:.1f}\%'.format(key))
                       for key, val in dict_color.items()]
    
    rate_elements = [Line2D([0], [0], marker=val, color='k', label='{:.5f}'.format(key), ls = 'None', \
                     markerfacecolor='None', markeredgewidth = 1.5, markersize=msize) for key, val in dict_marker.items()]
        
    temp_elements = []
    
    for key, val in dict_shade.items():
        cl = 'None'
        if val:
            cl = 'k'
        temp_elements.append(Line2D([0], [0], marker='h', color='k', label='\SI{%d}{\celsius}'%key, ls = 'None', \
             markerfacecolor=cl, markeredgewidth = 1.5, markersize=msize))
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    l1 = ax.legend(title = 'Strain Range', 
              handles=strain_elements,
              loc='center right',
              bbox_to_anchor=(1.4, 0.525),
              edgecolor = 'None')
    
    l2 = ax.legend(title = 'Strain Rate (/s)', 
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
    
    ax.set_title('$\chi^2 = %.3f$'%chi_ratio(y_pred, y_obs))
    
    ax.tick_params(axis = 'both', direction='in', which = 'both')
    
    if save:
        plt.savefig(os.path.join(path, save))
    
    plt.show()
        
def graph_nn_1_fold(data, log = False, load = True, save = '', which = 'both'):
    if load:
        print(data)
        d = np.load(data)
    else:
        d = data
    if which == 'test':
        y_obs, y_pred = d['y_obs_test'].reshape(11,-1), d['y_pred_test'].reshape(11,-1)
    elif which == 'train':
        y_obs, y_pred = d['y_obs_train'].reshape(33,-1), d['y_pred_train'].reshape(33,-1)
    elif which == 'both':
        y_obs = np.concatenate((d['y_obs_train'].reshape(33,-1), d['y_obs_test'].reshape(11,-1)), axis = 0)
        y_pred = np.concatenate((d['y_pred_train'].reshape(33,-1), d['y_pred_test'].reshape(11,-1)), axis = 0)
        # y_obs, y_pred = np.concatenate((d['y_obs_train'], d['y_obs_test'])), \
        #                 np.concatenate((d['y_pred_train'], d['y_pred_test']))
    
    strain_data = {}
    
    for test in fatigue_data.data:
        c = get_nf(test)
        for j, el in enumerate(np.rint(y_obs.flatten()).astype('int')):
            if c == el:
                strain_data.setdefault((test.Temp, test.Strain, test.Rate), [])
                strain_data[(test.Temp, test.Strain, test.Rate)].append((y_pred.flatten()[j], el))
                # break
    
    fig, ax = plt.subplots(figsize=(8,4))
    fig.subplots_adjust(right=0.5)
    
    ax.set_xlabel('Predicted $N_f$')
    ax.set_ylabel('Measured $N_f$')
        
    ax.set_ylim(100, 12000)
    ax.set_xlim(100, 12000)
    
    if log:
        ax.set_ylim(100, 2e4)
        ax.set_xlim(100, 2e4)
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    # ax.set_aspect('equal')
    ax.fill_between([100, 1e5], 100, [100, 1e5], color = 'k', alpha = 0.1)
    ax.plot([100, 1e5], [100, 1e5], lw = 2, color = 'k')
    
    if not log:
        ax.plot([0, 6000], [0, 12000], lw = 1, ls = '--', color = 'gray')
        ax.plot([0, 12000], [0, 6000], lw = 1, ls = '--', color = 'gray')
    else:
        ax.plot([100, 1e5], [200, 2e5], lw = 1, ls = '--', color = 'gray')
        ax.plot([200, 2e5], [100, 1e5], lw = 1, ls = '--', color = 'gray')
    
    msize = 7
    
    colors = plt.cm.gist_rainbow(np.linspace(0,1,6)).tolist()
    colors[1] = 'xkcd:orange'
    colors[2] = 'xkcd:green'
    colors[3] = 'xkcd:sky blue'
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
        
        ax.plot(x, y, marker = marker, markersize = msize, ls = 'None', \
        markeredgecolor = color, markerfacecolor = facecol, markeredgewidth = 1.5)
            
    strain_elements = [Patch(facecolor= val, edgecolor=val, label='{:.1f}\%'.format(key))
                       for key, val in dict_color.items()]
    
    rate_elements = [Line2D([0], [0], marker=val, color='k', label='{:.5f}'.format(key), ls = 'None', \
                     markerfacecolor='None', markeredgewidth = 1.5, markersize=msize) for key, val in dict_marker.items()]
        
    temp_elements = []
    
    for key, val in dict_shade.items():
        cl = 'None'
        if val:
            cl = 'k'
        temp_elements.append(Line2D([0], [0], marker='h', color='k', label='\SI{%d}{\celsius}'%key, ls = 'None', \
             markerfacecolor=cl, markeredgewidth = 1.5, markersize=msize))
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    
        # bta = [(1, 0.2), (0.01, 0.83), (0.01, 1)]
    fs = 9
    bta = [(1, 0.25), (0.01, 1), (0.01, 0.73)]
    
    l1 = ax.legend(title = 'Strain Range', 
              handles=strain_elements,
              loc='center right',
              bbox_to_anchor=bta[0],
              edgecolor = 'k',
              facecolor = '#e6e6e6',
              framealpha = 1,
              fontsize = fs)
    
    l2 = ax.legend(title = 'Strain Rate [s$^{-1}$]', 
          handles=rate_elements,
          loc='upper left',
          bbox_to_anchor=bta[1],
          fontsize = fs,
          framealpha = 1,
          edgecolor = 'k')
    
    l3 = ax.legend(title = 'Temperature', 
          handles=temp_elements,
          loc='upper left',
          bbox_to_anchor=bta[2],
          fontsize = fs,
          framealpha = 1,
          edgecolor = 'k')
    
    l2._legend_box.align = "left"
    l3._legend_box.align = "left"
    
    ax.add_artist(l1)
    ax.add_artist(l2)
    ax.add_artist(l3)
    
    
    ax.set_title('$\chi^2 = %.3f$'%chi_ratio(y_pred, y_obs))
    
    ax.tick_params(axis = 'both', direction='in', which = 'both')
    
    if save:
        plt.savefig(os.path.join(path, save))
    
    plt.show()

def graph_nn_hist(data, log = False, load = True, save = '', which = 'both', bins = 10):
    if load:
        print(data)
        d = np.load(data)
    else:
        d = data
    if which == 'test':
        y_obs, y_pred = d['y_obs_test'].reshape(11,-1), d['y_pred_test'].reshape(11,-1)
    elif which == 'train':
        y_obs, y_pred = d['y_obs_train'].reshape(33,-1), d['y_pred_train'].reshape(33,-1)
    elif which == 'both':
        y_obs = np.concatenate((d['y_obs_train'].reshape(33,-1), d['y_obs_test'].reshape(11,-1)), axis = 0)
        y_pred = np.concatenate((d['y_pred_train'].reshape(33,-1), d['y_pred_test'].reshape(11,-1)), axis = 0)
    
    strain_data = {}
    
    for test in fatigue_data.data:
        c = get_nf(test)
        for j, el in enumerate(np.rint(y_obs.flatten()).astype('int')):
            if c == el:
                strain_data.setdefault((test.Temp, test.Strain, test.Rate), [])
                strain_data[(test.Temp, test.Strain, test.Rate)].append((y_pred.flatten()[j], el))
                # break
    
    fig, ax = plt.subplots(figsize=(5,5))
    fig.subplots_adjust(right=0.7, top = 0.8)
    
    ax.set_xlabel('Predicted $N_f$')
    ax.set_ylabel('Measured $N_f$')
        
    ax.set_ylim(100, 12000)
    ax.set_xlim(100, 12000)
    
    if log:
        ax.set_ylim(100, 1e5)
        ax.set_xlim(100, 1e5)
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    ax.set_aspect('equal')
    ax.fill_between([100, 1e5], 100, [100, 1e5], color = '#e6e6e6')
    ax.plot([100, 1e5], [100, 1e5], lw = 2, color = 'k')
    
    if not log:
        ax.plot([0, 6000], [0, 12000], lw = 1, ls = '--', color = 'gray')
        ax.plot([0, 12000], [0, 6000], lw = 1, ls = '--', color = 'gray')
    else:
        ax.plot([100, 1e5], [200, 2e5], lw = 1, ls = '--', color = 'gray')
        ax.plot([200, 2e5], [100, 1e5], lw = 1, ls = '--', color = 'gray')
    
    msize = 7
    
    colors = plt.cm.gist_rainbow(np.linspace(0,1,6)).tolist()
    colors[1] = 'xkcd:orange'
    colors[2] = 'xkcd:green'
    colors[3] = 'xkcd:sky blue'
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
        
        ax.plot(x, y, marker = marker, markersize = msize, ls = 'None', \
        markeredgecolor = color, markerfacecolor = facecol, markeredgewidth = 1.5)
            
    strain_elements = [Patch(facecolor= val, edgecolor=val, label='{:.1f}\%'.format(key))
                       for key, val in dict_color.items()]
    
    rate_elements = [Line2D([0], [0], marker=val, color='k', label='{:.5f}'.format(key), ls = 'None', \
                     markerfacecolor='None', markeredgewidth = 1.5, markersize=msize) for key, val in dict_marker.items()]
        
    temp_elements = []
    
    for key, val in dict_shade.items():
        cl = 'None'
        if val:
            cl = 'k'
        temp_elements.append(Line2D([0], [0], marker='h', color='k', label='\SI{%d}{\celsius}'%key, ls = 'None', \
             markerfacecolor=cl, markeredgewidth = 1.3, markersize=msize))
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    fs = 9
    
    l1 = ax.legend(title = 'Strain Range', 
              handles=strain_elements,
              loc='center right',
              bbox_to_anchor=(1, 0.25),
              edgecolor = 'None',
              facecolor = '#e6e6e6',
              framealpha = 1,
              fontsize = fs)
    
    l2 = ax.legend(title = 'Strain Rate [s$^{-1}$]', 
          handles=rate_elements,
          loc='upper left',
          bbox_to_anchor=(0.01, 0.77),
          fontsize = fs,
          framealpha = 1,
          edgecolor = 'None')
    
    l3 = ax.legend(title = 'Temperature', 
          handles=temp_elements,
          loc='upper left',
          bbox_to_anchor=(0.015, 1),
          fontsize = fs,
          framealpha = 1,
          edgecolor = 'None')
    
    l2._legend_box.align = "left"
    l3._legend_box.align = "left"
    
    ax.add_artist(l1)
    ax.add_artist(l2)
    ax.add_artist(l3)
    
    ax.set_title('$\chi^2 = %.3f$'%chi_ratio(y_pred, y_obs))
    
    ax.tick_params(axis = 'both', direction='in', which = 'both')
    
    a = ax.inset_axes([.8, .8, .25, .25], facecolor='#d1d1e0')
    
    y_diff = y_pred - y_obs
    
    (n, bins, patches) = a.hist(y_diff, bins = bins, color = 'k',edgecolor = '#d1d1e0')
    
    a.set_xlim(-2e3, 2e3)
    
    a.tick_params(axis='both', which='both', direction='in', bottom=False, left = False, labelleft=False,\
                  labelbottom=False, labeltop=True)    
    a.set_xticklabels([r'$-2\cdot10^3$', '0', r'$2\cdot10^3$'])

    a.text(0.9, 0.9, '%d'%max(n), ha = 'right', va = 'top', transform = a.transAxes)
    
    if save:
        plt.savefig(os.path.join(path, save))
    
    plt.show()

def graph_nn_hist_only(data, bin_width = 5, load = True, save = '', which = 'both', ax = None, v2 = True):
    
    if load:
        print(data)
        d = np.load(data)
    else:
        d = data
    if which == 'test':
        y_obs, y_pred = d['y_obs_test'].reshape(11,-1), d['y_pred_test'].reshape(11,-1)
    elif which == 'dev':
        y_obs, y_pred = d['y_obs_dev'].reshape(11,-1), d['y_pred_dev'].reshape(11,-1)
    elif which == 'train':
        y_obs, y_pred = d['y_obs_train'].reshape(22,-1), d['y_pred_train'].reshape(22,-1)
    elif which == 'all':
        if v2:
            y_obs = np.concatenate((d['y_obs_train'].reshape(22,-1),
                        d['y_obs_dev'].reshape(11,-1),
                        d['y_obs_test'].reshape(11,-1)), axis = 0)
            y_pred = np.concatenate((d['y_pred_train'].reshape(22,-1),
                                     d['y_pred_dev'].reshape(11,-1),
                                     d['y_pred_test'].reshape(11,-1)), axis = 0)
        else:
            y_obs = np.concatenate((d['y_obs_train'].reshape(33,-1), d['y_obs_test'].reshape(11,-1)), axis = 0)
            y_pred = np.concatenate((d['y_pred_train'].reshape(33,-1), d['y_pred_test'].reshape(11,-1)), axis = 0)
        
    y_diff = (y_pred - y_obs)/y_obs*100
    y_diff = y_diff.flatten()
    
    
    if not ax:
        _, ax = plt.subplots(figsize=(4,4))
    else:
        fs = 12
        ax.xaxis.label.set_size(fs)
        ax.yaxis.label.set_size(fs)
        ax.tick_params(axis='both', which='major', labelsize=11)
    
    
    ax.fill_between([0, 100], 5, color = 'k', alpha = 0.1)
    ax.plot([0, 0], [0, 5], lw = 2, color = 'k')
    
    # ax.plot([50, 50], [0, 5], lw = 1, ls = '--', color = 'gray')
     
    bins = range(-100, 100+bin_width, bin_width)
    
    ax.hist(np.clip(y_diff, -100, 99), bins = bins, color = '#595959', ec="black", alpha = 1, weights=np.ones(len(y_diff)) / len(y_diff))
    
    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 0.5)
    # if max(abs(y_diff)) > 100:
    #     ax.set_xlim(-200, 200)
    # if max(abs(y_diff)) > 200:
    #     ax.set_xlim(-300, 300)
    
    ax.grid(dashes = (1, 5), color = 'k', lw = 0.7)
    
    ax.set_xlabel('Percentage Error [\%]')
    ax.set_ylabel('Frequency')
    
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    
    if save:
        plt.savefig(os.path.join(path, save), bbox_inches = 'tight')
    
    

def graph_nn_2_fold(data, log = False, load = True, save = '', which = 'both'):
    if load:
        print(data)
        d = np.load(data)
    else:
        d = data
    
    if which == 'test':
        y_obs, y_pred = d['y_obs_test'].reshape(11,-1), d['y_pred_test'].reshape(11,-1)
    elif which == 'train':
        y_obs, y_pred = d['y_obs_train'].reshape(33,-1), d['y_pred_train'].reshape(33,-1)
    elif which == 'both':
        y_obs = np.concatenate((d['y_obs_train'].reshape(33,-1), d['y_obs_test'].reshape(11,-1)), axis = 0)
        y_pred = np.concatenate((d['y_pred_train'].reshape(33,-1), d['y_pred_test'].reshape(11,-1)), axis = 0)
    
    strain_data = {}
    
    for test in fatigue_data.data:
        c = get_nf(test)
        for j, el in enumerate(np.rint(y_obs.flatten()).astype('int')):
            if c == el:
                if el in np.rint(y_obs):
                    cross = 'train'
                else:
                    cross = 'test'
                strain_data.setdefault((test.Temp, test.Strain, test.Rate, cross), [])
                strain_data[(test.Temp, test.Strain, test.Rate, cross)].append((y_pred.flatten()[j], el))
    
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
    ax.fill_between([100, 20000], 20000, [100, 20000], color = '#ffffe6')
    ax.fill_between([100, 20000], 100, [100, 20000], color = '#e6f3ff')
    ax.plot([100, 20000], [100, 20000], lw = 2, color = 'k')
    
    if not log:
        ax.plot([0, 6000], [0, 12000], lw = 1, ls = '--', color = 'gray')
        ax.plot([0, 12000], [0, 6000], lw = 1, ls = '--', color = 'gray')
    else:
        ax.plot([100, 20000], [200, 40000], lw = 1, ls = '--', color = 'gray')
        ax.plot([200, 40000], [100, 20000], lw = 1, ls = '--', color = 'gray')
    
    msize = 9
    
    colors = plt.cm.gist_rainbow(np.linspace(0,1,6)).tolist()
    colors[1] = 'xkcd:orange'
    colors[2] = 'xkcd:green'
    colors[3] = 'xkcd:sky blue'
    colors[4] = '#0066ff'
    strain_vals = [0.3, 0.4, 0.6, 1, 2, 3]
    
    dict_color = dict(zip(strain_vals, colors))
    dict_marker = {0.001 : 'o', 0.0001: '^', 1e-5: 's'}
    dict_shade = {850 : False, 950: True}
    dict_cross = {'train' : False, 'test': True}
    
    for key, value in strain_data.items():
        temp, strain, rate, cross = key
        color, marker, shade, cross = dict_color[strain], dict_marker[rate], dict_shade[temp], dict_cross[cross]
        
        if shade:
            facecol = color
        else:
            facecol = 'None'
        
        x, y = zip(*value)
        
        ax.plot(x, y, marker = marker, markersize = msize, ls = 'None', \
        markeredgecolor = color, markerfacecolor = facecol, markeredgewidth = 1.5)
        if cross:
            ax.plot(x, y, marker = '.', markersize = 4, ls = 'None', color = 'k')
            
            
    strain_elements = [Patch(facecolor= val, edgecolor=val, label='{:.1f}\%'.format(key))
                       for key, val in dict_color.items()]
    
    rate_elements = [Line2D([0], [0], marker=val, color='k', label='{:.5f}'.format(key), ls = 'None', \
                     markerfacecolor='None', markeredgewidth = 1.5, markersize=msize) for key, val in dict_marker.items()]
        
    temp_elements = []
    
    for key, val in dict_shade.items():
        cl = 'None'
        if val:
            cl = 'k'
        temp_elements.append(Line2D([0], [0], marker='h', color='k', label='\SI{%d}{\celsius}'%key, ls = 'None', \
             markerfacecolor=cl, markeredgewidth = 1.5, markersize=msize))
    
    ax.grid(dashes = (1, 5), color = 'k', lw = 0.7)
    
    l1 = ax.legend(title = 'Strain Range', 
              handles=strain_elements,
              loc='center right',
              bbox_to_anchor=(1.4, 0.525),
              edgecolor = 'None')
    
    l2 = ax.legend(title = 'Strain Rate (/s)', 
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
    
    ax.set_title('$\chi^2 = %.3f$'%chi_ratio(y_pred, y_obs))
    
    ax.tick_params(axis = 'both', direction='in', which = 'both')
    
    if save:
        plt.savefig(os.path.join(path, save))
    
    plt.show()
    
def get_meap(data, load = True, which = 'all', v2 = True):
    if load:
        d = np.load(data)
    else:
        d = data
    if which == 'test':
        y_obs, y_pred = d['y_obs_test'].reshape(11,-1), d['y_pred_test'].reshape(11,-1)
    if which == 'dev':
        y_obs, y_pred = d['y_obs_dev'].reshape(11,-1), d['y_pred_dev'].reshape(11,-1)
    elif which == 'train':
        if v2:
            y_obs, y_pred = d['y_obs_train'].reshape(22,-1), d['y_pred_train'].reshape(22,-1)
        else:
            y_obs, y_pred = d['y_obs_train'].reshape(33,-1), d['y_pred_train'].reshape(33,-1)
    elif which == 'all':
        if v2:
            y_obs = np.concatenate((d['y_obs_train'].reshape(22,-1),
                        d['y_obs_dev'].reshape(11,-1),
                        d['y_obs_test'].reshape(11,-1)), axis = 0)
            y_pred = np.concatenate((d['y_pred_train'].reshape(22,-1),
                                     d['y_pred_dev'].reshape(11,-1),
                                     d['y_pred_test'].reshape(11,-1)), axis = 0)
        else:
            y_obs = np.concatenate((d['y_obs_train'].reshape(33,-1), d['y_obs_test'].reshape(11,-1)), axis = 0)
            y_pred = np.concatenate((d['y_pred_train'].reshape(33,-1), d['y_pred_test'].reshape(11,-1)), axis = 0)
    
    return (abs(y_obs-y_pred)/y_obs).mean()*100

def get_chi(data, load = True, which = 'all', v2 = True):
    if load:
        d = np.load(data)
    else:
        d = data
    if which == 'test':
        y_obs, y_pred = d['y_obs_test'].reshape(11,-1), d['y_pred_test'].reshape(11,-1)
    if which == 'dev':
        y_obs, y_pred = d['y_obs_dev'].reshape(11,-1), d['y_pred_dev'].reshape(11,-1)
    elif which == 'train':
        if v2:
            y_obs, y_pred = d['y_obs_train'].reshape(22,-1), d['y_pred_train'].reshape(22,-1)
        else:
            y_obs, y_pred = d['y_obs_train'].reshape(33,-1), d['y_pred_train'].reshape(33,-1)
    elif which == 'all':
        if v2:
            y_obs = np.concatenate((d['y_obs_train'].reshape(22,-1),
                        d['y_obs_dev'].reshape(11,-1),
                        d['y_obs_test'].reshape(11,-1)), axis = 0)
            y_pred = np.concatenate((d['y_pred_train'].reshape(22,-1),
                                     d['y_pred_dev'].reshape(11,-1),
                                     d['y_pred_test'].reshape(11,-1)), axis = 0)
        else:
            y_obs = np.concatenate((d['y_obs_train'].reshape(33,-1), d['y_obs_test'].reshape(11,-1)), axis = 0)
            y_pred = np.concatenate((d['y_pred_train'].reshape(33,-1), d['y_pred_test'].reshape(11,-1)), axis = 0)
    
    return chi_ratio(y_pred, y_obs)

def graph_nn_1_dev(data, log = False, load = True, save = '', which = 'all'):
    if load:
        print(data)
        d = np.load(data)
    else:
        d = data
    if which == 'test':
        y_obs, y_pred = d['y_obs_test'].reshape(11,-1), d['y_pred_test'].reshape(11,-1)
    elif which == 'dev':
        y_obs, y_pred = d['y_obs_dev'].reshape(11,-1), d['y_pred_dev'].reshape(11,-1)
    elif which == 'train':
        y_obs, y_pred = d['y_obs_train'].reshape(22,-1), d['y_pred_train'].reshape(22,-1)
    elif which == 'all':
        y_obs = np.concatenate((d['y_obs_train'].reshape(22,-1),
                                d['y_obs_dev'].reshape(11,-1),
                                d['y_obs_test'].reshape(11,-1)), axis = 0)
        y_pred = np.concatenate((d['y_pred_train'].reshape(22,-1),
                                 d['y_pred_dev'].reshape(11,-1),
                                 d['y_pred_test'].reshape(11,-1)), axis = 0)
    strain_data = {}
    
    for test in fatigue_data.data:
        c = get_nf(test)
        for j, el in enumerate(np.rint(y_obs.flatten()).astype('int')):
            if c == el:
                strain_data.setdefault((test.Temp, test.Strain, test.Rate), [])
                strain_data[(test.Temp, test.Strain, test.Rate)].append((y_pred.flatten()[j], el))
                # break
    
    fig, ax = plt.subplots(figsize=(8,4))
    fig.subplots_adjust(right=0.5)
    
    ax.set_xlabel('Predicted $N_f$')
    ax.set_ylabel('Measured $N_f$')
        
    ax.set_ylim(100, 12000)
    ax.set_xlim(100, 12000)
    
    if log:
        ax.set_ylim(100, 2e4)
        ax.set_xlim(100, 2e4)
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    # ax.set_aspect('equal')
    ax.fill_between([100, 1e5], 100, [100, 1e5], color = 'k', alpha = 0.1)
    ax.plot([100, 1e5], [100, 1e5], lw = 2, color = 'k')
    
    if not log:
        ax.plot([0, 6000], [0, 12000], lw = 1, ls = '--', color = 'gray')
        ax.plot([0, 12000], [0, 6000], lw = 1, ls = '--', color = 'gray')
    else:
        ax.plot([100, 1e5], [200, 2e5], lw = 1, ls = '--', color = 'gray')
        ax.plot([200, 2e5], [100, 1e5], lw = 1, ls = '--', color = 'gray')
    
    msize = 7
    
    colors = plt.cm.gist_rainbow(np.linspace(0,1,6)).tolist()
    colors[1] = 'xkcd:orange'
    colors[2] = 'xkcd:green'
    colors[3] = 'xkcd:sky blue'
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
        
        ax.plot(x, y, marker = marker, markersize = msize, ls = 'None', \
        markeredgecolor = color, markerfacecolor = facecol, markeredgewidth = 1.5)
            
    strain_elements = [Patch(facecolor= val, edgecolor=val, label='{:.1f}\%'.format(key))
                       for key, val in dict_color.items()]
    
    rate_elements = [Line2D([0], [0], marker=val, color='k', label='{:.5f}'.format(key), ls = 'None', \
                     markerfacecolor='None', markeredgewidth = 1.5, markersize=msize) for key, val in dict_marker.items()]
        
    temp_elements = []
    
    for key, val in dict_shade.items():
        cl = 'None'
        if val:
            cl = 'k'
        temp_elements.append(Line2D([0], [0], marker='h', color='k', label='\SI{%d}{\celsius}'%key, ls = 'None', \
             markerfacecolor=cl, markeredgewidth = 1.5, markersize=msize))
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    l1 = ax.legend(title = 'Strain Range', 
              handles=strain_elements,
              loc='center right',
              bbox_to_anchor=(1.4, 0.525),
              edgecolor = 'None')
    
    l2 = ax.legend(title = 'Strain Rate (/s)', 
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
    
    ax.set_title('$\chi^2 = %.3f$'%chi_ratio(y_pred, y_obs))
    
    ax.tick_params(axis = 'both', direction='in', which = 'both')
    
    if save:
        plt.savefig(os.path.join(path, save))
    
    plt.show()


def graph_nn_11_dev(data, log = False, load = True, save = '', which = 'all', ax = None, v2 = True):
    if load:
        print(data)
        d = np.load(data)
    else:
        d = data
    if which == 'test':
        y_obs, y_pred = d['y_obs_test'].reshape(11,-1), d['y_pred_test'].reshape(11,-1)
    elif which == 'dev':
        y_obs, y_pred = d['y_obs_dev'].reshape(11,-1), d['y_pred_dev'].reshape(11,-1)
    elif which == 'train':
        y_obs, y_pred = d['y_obs_train'].reshape(22,-1), d['y_pred_train'].reshape(22,-1)
    elif which == 'all':
        if not v2:
            y_obs = np.concatenate((d['y_obs_train'].reshape(33,-1),
                                    d['y_obs_test'].reshape(11,-1)), axis = 0)
            y_pred = np.concatenate((d['y_pred_train'].reshape(33,-1),
                                     d['y_pred_test'].reshape(11,-1)), axis = 0)
        else:
            y_obs = np.concatenate((d['y_obs_train'].reshape(22,-1),
                                    d['y_obs_dev'].reshape(11,-1),
                                    d['y_obs_test'].reshape(11,-1)), axis = 0)
            y_pred = np.concatenate((d['y_pred_train'].reshape(22,-1),
                                     d['y_pred_dev'].reshape(11,-1),
                                     d['y_pred_test'].reshape(11,-1)), axis = 0)
    strain_data = {}
    
    for test in fatigue_data.data:
        c = get_nf(test)
        for j, el in enumerate(np.rint(y_obs.flatten()).astype('int')):
            if c == el:
                strain_data.setdefault((test.Temp, test.Strain, test.Rate), [])
                strain_data[(test.Temp, test.Strain, test.Rate)].append((y_pred.flatten()[j], el))
                # break
    
    _plot = False
    if not ax:
        _plot = True
        fig, ax = plt.subplots(figsize=(8,4))
        fig.subplots_adjust(right=0.5)
    
    ax.set_xlabel('Predicted $N_f$')
    ax.set_ylabel('Measured $N_f$')
        
    ax.set_ylim(100, 12000)
    ax.set_xlim(100, 12000)
    
    if log:
        ax.set_ylim(100, 2e4)
        ax.set_xlim(100, 2e4)
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    # ax.set_aspect('equal')
    ax.fill_between([100, 1e5], 100, [100, 1e5], color = 'k', alpha = 0.1)
    ax.plot([100, 1e5], [100, 1e5], lw = 2, color = 'k')
    
    if not log:
        ax.plot([0, 6000], [0, 12000], lw = 1, ls = '--', color = 'gray')
        ax.plot([0, 12000], [0, 6000], lw = 1, ls = '--', color = 'gray')
    else:
        ax.plot([100, 1e5], [200, 2e5], lw = 1, ls = '--', color = 'gray')
        ax.plot([200, 2e5], [100, 1e5], lw = 1, ls = '--', color = 'gray')
    
    msize = 7
    
    fs = 13
    bta = [(0.99, 0.01), (0.01, 0.75), (0.01, 0.95)]
    if _plot:
        fs = 9
        bta = [(0.99, 0.01), (0.01, 0.99), (0.01, 0.72)]
    
    if not _plot:
        msize = 14
        ax.xaxis.label.set_size(fs)
        ax.yaxis.label.set_size(fs)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    colors = plt.cm.gist_rainbow(np.linspace(0,1,6)).tolist()
    
    
    colors[1] = 'xkcd:orange'
    colors[2] = '#7AC108'
    colors[4] = 'b'
    colors[3] = '#009DFF'
    colors[-1] = '#C108BA'
    
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
        
        ax.plot(x, y, marker = marker, markersize = msize, ls = 'None', \
        markeredgecolor = color, markerfacecolor = facecol, markeredgewidth = 2)
            
    strain_elements = [Patch(facecolor= val, edgecolor=val, label='{:.1f}\%'.format(key))
                       for key, val in dict_color.items()]
    
    rate_elements = [Line2D([0], [0], marker=val, color='k', label='{:.5f}'.format(key), ls = 'None', \
                     markerfacecolor='None', markeredgewidth = 2, markersize=msize) for key, val in dict_marker.items()]
        
    temp_elements = []
    
    for key, val in dict_shade.items():
        cl = 'None'
        if val:
            cl = 'k'
        temp_elements.append(Line2D([0], [0], marker='h', color='k', label='\SI{%d}{\celsius}'%key, ls = 'None', \
             markerfacecolor=cl, markeredgewidth = 2, markersize=msize))
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    l1 = ax.legend(title = 'Strain Range', 
              handles=strain_elements,
              loc='lower right',
              bbox_to_anchor=bta[0],
              edgecolor = 'k',
              facecolor = '#e6e6e6',
              framealpha = 1,
              fontsize = fs,
              title_fontsize = fs)
    
    l2 = ax.legend(title = 'Strain Rate [s$^{-1}$]', 
          handles=rate_elements,
          loc='upper left',
          bbox_to_anchor=bta[1],
          fontsize = fs,
          title_fontsize = fs,
          framealpha = 1,
          edgecolor = 'k')
    
    l3 = ax.legend(title = 'Temperature', 
          handles=temp_elements,
          loc='upper left',
          bbox_to_anchor=bta[2],
          fontsize = fs,
          title_fontsize = fs,
          framealpha = 1,
          edgecolor = 'k')
    
    l2._legend_box.align = "left"
    l3._legend_box.align = "left"
    
    ax.add_artist(l1)
    ax.add_artist(l2)
    ax.add_artist(l3)
    
    # ax.set_title('$\chi^2 = %.3f$'%chi_ratio(y_pred, y_obs))
    
    ax.tick_params(axis = 'both', direction='in', which = 'both')
    
    if save:
        plt.savefig(os.path.join(path, save))
    
    if _plot:
        plt.show()


def graph_nn_1m_dev(data, log = False, load = True, save = '', which = 'all', ax = None, ax_loc = 'normal', v2 = True, ver = False):
    if load:
        print(data)
        d = np.load(data)
    else:
        d = data
    if which == 'test':
        y_obs, y_pred = d['y_obs_test'].reshape(11,-1), d['y_pred_test'].reshape(11,-1)
    elif which == 'dev':
        y_obs, y_pred = d['y_obs_dev'].reshape(11,-1), d['y_pred_dev'].reshape(11,-1)
    elif which == 'train':
        y_obs, y_pred = d['y_obs_train'].reshape(22,-1), d['y_pred_train'].reshape(22,-1)
    elif which == 'all':
        if not v2:
            y_obs = np.concatenate((d['y_obs_train'].reshape(33,-1),
                                    d['y_obs_test'].reshape(11,-1)), axis = 0)
            y_pred = np.concatenate((d['y_pred_train'].reshape(33,-1),
                                     d['y_pred_test'].reshape(11,-1)), axis = 0)
        else:
            y_obs = np.concatenate((d['y_obs_train'].reshape(22,-1),
                                    d['y_obs_dev'].reshape(11,-1),
                                    d['y_obs_test'].reshape(11,-1)), axis = 0)
            y_pred = np.concatenate((d['y_pred_train'].reshape(22,-1),
                                     d['y_pred_dev'].reshape(11,-1),
                                     d['y_pred_test'].reshape(11,-1)), axis = 0)
    trial_data = {}
    
    for test in fatigue_data.data:
        c = get_nf(test)
        for j, el in enumerate(np.rint(y_obs.flatten()).astype('int')):
            if c == el:
                trial_data.setdefault(c, [])
                trial_data[c].append(y_pred.flatten()[j])
                
                
    _plot = False
    if not ax:
        _plot = True
        fig, ax = plt.subplots(figsize=(8,4))
        fig.subplots_adjust(right=0.5)
    
    ax.set_xlabel('Predicted $N_f$')
    ax.set_ylabel('Measured $N_f$')
        
    ax.set_ylim(100, 12000)
    ax.set_xlim(100, 12000)
    
    if log:
        ax.set_ylim(100, 2e4)
        ax.set_xlim(100, 2e4)
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    if _plot:
        # ax.set_aspect('equal')
        ax.fill_between([100, 1e5], 100, [100, 1e5], color = 'k', alpha = 0.1)
        ax.plot([100, 1e5], [100, 1e5], lw = 2, color = 'k')
        
        if not log:
            ax.plot([0, 6000], [0, 12000], lw = 1, ls = '--', color = 'gray')
            ax.plot([0, 12000], [0, 6000], lw = 1, ls = '--', color = 'gray')
        else:
            ax.plot([100, 1e5], [200, 2e5], lw = 1, ls = '--', color = 'gray')
            ax.plot([200, 2e5], [100, 1e5], lw = 1, ls = '--', color = 'gray')
    
    msize = 7
    fs = 10
    if ax_loc == 'minor':
            msize = 4
            fs = 8
    elif ax_loc == 'major':
        msize = 10
        fs = 13
    
    for key, value in trial_data.items():
        if key in np.rint(d['y_obs_train']).astype('int'):
            label = 'train'
        elif key in np.rint(d['y_obs_test']).astype('int'):
            label = 'test'
        else:
            label = 'dev'
            
        trial_data[key] = [value, np.mean(value), label]

    y = trial_data.keys()
    v = list(trial_data.values())
    x = [a[1] for a in v]

    ax.plot(x, y, marker = 'o', markersize = msize, ls = 'None', \
    markeredgecolor = 'k', markerfacecolor = '#595959', markeredgewidth = 2)
            
    ax.tick_params(axis = 'both', direction='in', which = 'both')
    
    bta = (0.72, 0.01) if ax_loc == 'major' else (0.99, 0.01)
    
    LEG = [Line2D([0], [0], marker = 'o', markersize = msize, ls = 'None', \
    markeredgecolor = 'k', markerfacecolor = '#595959', markeredgewidth = 2, label = 'Averaged\nPredictions')]
        
    l1 = ax.legend(handles=LEG,
              loc='lower right',
              bbox_to_anchor=bta,
              edgecolor = 'k',
              facecolor = '#e6e6e6',
              framealpha = 1,
              fontsize = fs)
    
    ax.add_artist(l1)
    
    if save:
        plt.savefig(os.path.join(path, save))
    
    if _plot:
        plt.show()
    
    x0 = np.array([(k, v[1]) for k, v in trial_data.items() if v[2] == 'train'])
    x1 = np.array([(k, v[1]) for k, v in trial_data.items() if v[2] == 'dev'])
    x2 = np.array([(k, v[1]) for k, v in trial_data.items() if v[2] == 'test'])
    
    m_data = {'y_obs_test': x2[:,0], 'y_pred_test': x2[:,1],
              'y_obs_train': x0[:,0], 'y_pred_train': x0[:,1],
              'y_obs_dev': x1[:,0], 'y_pred_dev':x1[:,1],}
    
    if ver:
        print('\n', get_meap(m_data, which = 'train', load = False))
        print(get_meap(m_data, which = 'dev', load = False))
        print(get_meap(m_data, which = 'test', load = False), '\n')
        
    return trial_data, m_data

def graph_nn_2_dev(data, log = False, load = True, save = ''):
    if load:
        print(data)
        d = np.load(data)
    else:
        d = data

    y_obs = np.concatenate((d['y_obs_train'].reshape(22,-1),
                        d['y_obs_dev'].reshape(11,-1),
                        d['y_obs_test'].reshape(11,-1)), axis = 0)
    y_pred = np.concatenate((d['y_pred_train'].reshape(22,-1),
                             d['y_pred_dev'].reshape(11,-1),
                             d['y_pred_test'].reshape(11,-1)), axis = 0)

    fig, ax = plt.subplots(figsize=(8,4))
    fig.subplots_adjust(right=0.5)
    
    ax.set_xlabel('Predicted $N_f$')
    ax.set_ylabel('Measured $N_f$')
        
    ax.set_ylim(100, 12000)
    ax.set_xlim(100, 12000)
    
    if log:
        ax.set_ylim(100, 2e4)
        ax.set_xlim(100, 2e4)
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    # ax.set_aspect('equal')
    ax.fill_between([100, 1e5], 100, [100, 1e5], color = 'k', alpha = 0.1)
    ax.plot([100, 1e5], [100, 1e5], lw = 2, color = 'k')
    
    if not log:
        ax.plot([0, 6000], [0, 12000], lw = 1, ls = '--', color = 'gray')
        ax.plot([0, 12000], [0, 6000], lw = 1, ls = '--', color = 'gray')
    else:
        ax.plot([100, 1e5], [200, 2e5], lw = 1, ls = '--', color = 'gray')
        ax.plot([200, 2e5], [100, 1e5], lw = 1, ls = '--', color = 'gray')
    
    msize = 7
    
    ax.plot(d['y_pred_train'], d['y_obs_train'], marker = 'o', markersize = msize, ls = 'None', \
        markeredgecolor = '#ff6600', markerfacecolor = '#ff6600', markeredgewidth = 1.5, label = 'Train'
        + '%.1f'%get_meap(data, load, 'train') + r'\%')
    ax.plot(d['y_pred_dev'], d['y_obs_dev'], marker = 's', markersize = msize, ls = 'None', \
        markeredgecolor = '#ff66ff', markerfacecolor = '#ff66ff', markeredgewidth = 1.5, label = 'Dev'
        + '%.1f'%get_meap(data, load, 'dev') + r'\%')
    ax.plot(d['y_pred_test'], d['y_obs_test'], marker = '^', markersize = msize+2, ls = 'None', \
        markeredgecolor = '#29a329', markerfacecolor = '#29a329', markeredgewidth = 1.5, label = 'Test'
        + '%.1f'%get_meap(data, load, 'test') + r'\%')

    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    ax.legend(loc='center right',
      bbox_to_anchor=(1.35, 0.55),
      edgecolor = 'None')
    
    ax.set_title('$\chi^2 = %.3f$'%chi_ratio(y_pred, y_obs))
    
    ax.tick_params(axis = 'both', direction='in', which = 'both')
    
    if save:
        plt.savefig(os.path.join(path, save))
    
    plt.show()

def graph_nn_22_dev(data, log = False, load = True, save = '', ax = None, title = None):
    if load:
        print(data)
        d = np.load(data)
    else:
        d = data

    y_obs = np.concatenate((d['y_obs_train'].reshape(22,-1),
                        d['y_obs_dev'].reshape(11,-1),
                        d['y_obs_test'].reshape(11,-1)), axis = 0)
    y_pred = np.concatenate((d['y_pred_train'].reshape(22,-1),
                             d['y_pred_dev'].reshape(11,-1),
                             d['y_pred_test'].reshape(11,-1)), axis = 0)
    
    _plot = False
    if not ax:
        _plot = True
        fig, ax = plt.subplots(figsize=(8,4))
        fig.subplots_adjust(right=0.5)
    
    ax.set_xlabel('Predicted $N_f$')
    ax.set_ylabel('Measured $N_f$')
        
    ax.set_ylim(100, 12000)
    ax.set_xlim(100, 12000)
    
    if log:
        ax.set_ylim(100, 2e4)
        ax.set_xlim(100, 2e4)
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    # ax.set_aspect('equal')
    ax.fill_between([100, 1e5], 100, [100, 1e5], color = 'k', alpha = 0.1)
    ax.plot([100, 1e5], [100, 1e5], lw = 2, color = 'k')
    
    if not log:
        ax.plot([0, 6000], [0, 12000], lw = 1, ls = '--', color = 'gray')
        ax.plot([0, 12000], [0, 6000], lw = 1, ls = '--', color = 'gray')
    else:
        ax.plot([100, 1e5], [200, 2e5], lw = 1, ls = '--', color = 'gray')
        ax.plot([200, 2e5], [100, 1e5], lw = 1, ls = '--', color = 'gray')
    
    msize = 5
    Labels = ['Train', 'Dev', 'Test']
    labels = ['Train', 'Dev', 'Test']
    
    if not _plot:
        msize = 3
        fs = 12
        ax.xaxis.label.set_size(fs)
        ax.yaxis.label.set_size(fs)
        ax.tick_params(axis='both', which='major', labelsize=11)
    else:
        labels = ['%s -- %.1f'%(s, get_meap(data, load, s.lower())) + r'\%' for s in labels]
    

    markers = ['x', 'o', 's']
    colors = ['#8000ff', '#ff1ac6', '#00b300']


    for i in range(3):
        l = Labels[i].lower()
        ax.plot(d[f'y_pred_{l}'], d[f'y_obs_{l}'], markers[i], markersize = msize+2, ls = 'None', \
            markerfacecolor = 'None', markeredgecolor = colors[i], markeredgewidth = 2, label = labels[i])

    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    if not _plot:
        l = ax.legend(loc='upper left', edgecolor = 'k', framealpha = 1, fontsize = 10)
    else:
        l = ax.legend(loc='upper left', edgecolor = 'k', framealpha = 1)
        props = dict(boxstyle='round', facecolor= (0.9, 0.9, 0.9), lw = 1)
        ax.text(0.95, 0.05, r'\textit{Non-Conservative Region}', transform=ax.transAxes,\
                va='bottom', ha = 'right', bbox=props)
    
    ax.add_artist(l)
    
    if title:
        ax.set_title(r'\textbf{%s}'%title)
    
    ax.tick_params(axis = 'both', direction='in', which = 'both')
    
    if save:
        plt.savefig(os.path.join(path, save))
    
    if _plot:
        plt.show()
    
def graph_nn_12_dev(data, log = False, load = True, save = '', which = 'all', avg = False):
    X = 4.5; S1 = 0.5; S2 = 0.6; 
    H = (X-S2)/2
    Y = (3*X+S1)/2
    
    fig = plt.figure(figsize=(X,Y))
    ax_main = fig.add_axes([0, (H+S1)/Y, 1, X/Y])
    ax_hist = fig.add_axes([0, 0, H/X, H/Y])
    ax_data = fig.add_axes([(H+S2)/X, 0, H/X, H/Y])
    
    # bins = 10 if len(y_obs.flatten()) <= 44 else 40
    bw = 10
    
    if avg:
        graph_nn_11_dev(data, log, load, '', which, ax_main)
        _, h_data = graph_nn_1m_dev(data, log, load, '', which, ax_main, 'major')
        graph_nn_22_dev(data, log, load, '', ax_data)
        graph_nn_1m_dev(data, log, load, '', which, ax_data, 'minor')
        graph_nn_hist_only(h_data, bw, False, '', which, ax_hist)
    else:
        graph_nn_11_dev(data, log, load, '', which, ax_main)
        graph_nn_22_dev(data, log, load, '', ax_data)
        graph_nn_hist_only(data, bw, load, '', which, ax_hist)

    if save:
        plt.savefig(os.path.join(path, save), bbox_inches = 'tight')

    plt.show()
    
def graph_test_train(data, log = False, load = True, save = '', which = 'all'):
    
    if load:
        print(data)
        d = np.load(data)
    else:
        d = data
    if which == 'test':
        y_obs, y_pred = d['y_obs_test'].reshape(11,-1), d['y_pred_test'].reshape(11,-1)
    elif which == 'train':
        y_obs, y_pred = d['y_obs_train'].reshape(33,-1), d['y_pred_train'].reshape(33,-1)
    elif which == 'all':
        y_obs = np.concatenate((d['y_obs_train'].reshape(33,-1),
                                d['y_obs_test'].reshape(11,-1)), axis = 0)
        y_pred = np.concatenate((d['y_pred_train'].reshape(33,-1),
                                 d['y_pred_test'].reshape(11,-1)), axis = 0)
   
    strain_data = {}
    
    for test in fatigue_data.data:
        c = get_nf(test)
        for j, el in enumerate(np.rint(y_obs.flatten()).astype('int')):
            if c == el:
                strain_data.setdefault((test.Temp, test.Strain, test.Rate), [])
                strain_data[(test.Temp, test.Strain, test.Rate)].append((y_pred.flatten()[j], el))
                # break
    
    fig, ax = plt.subplots(figsize=(8,4))
    fig.subplots_adjust(right=0.5)
    
    ax.set_xlabel('Predicted $N_f$')
    ax.set_ylabel('Measured $N_f$')
        
    ax.set_ylim(100, 12000)
    ax.set_xlim(100, 12000)
    
    if log:
        ax.set_ylim(100, 2e4)
        ax.set_xlim(100, 2e4)
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    # ax.set_aspect('equal')
    ax.fill_between([100, 1e5], 100, [100, 1e5], color = 'k', alpha = 0.1)
    ax.plot([100, 1e5], [100, 1e5], lw = 2, color = 'k')
    
    if not log:
        ax.plot([0, 6000], [0, 12000], lw = 1, ls = '--', color = 'gray')
        ax.plot([0, 12000], [0, 6000], lw = 1, ls = '--', color = 'gray')
    else:
        ax.plot([100, 1e5], [200, 2e5], lw = 1, ls = '--', color = 'gray')
        ax.plot([200, 2e5], [100, 1e5], lw = 1, ls = '--', color = 'gray')
    
    msize = 7
    
    colors = plt.cm.gist_rainbow(np.linspace(0,1,6)).tolist()
    
    colors[1] = 'xkcd:orange'
    colors[2] = '#7AC108'
    colors[4] = 'b'
    colors[3] = '#009DFF'
    colors[-1] = '#C108BA'
    
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
        
        ax.plot(x, y, marker = 'o', markersize = msize, ls = 'None', \
        markeredgecolor = color, markerfacecolor = color, markeredgewidth = 1)
            
    strain_elements = [Patch(facecolor= val, edgecolor=val, label='{:.1f}\%'.format(key))
                       for key, val in dict_color.items()]
    
    rate_elements = [Line2D([0], [0], marker=val, color='k', label='{:.5f}'.format(key), ls = 'None', \
                     markerfacecolor='None', markeredgewidth = 2, markersize=msize) for key, val in dict_marker.items()]
        
    temp_elements = []
    
    for key, val in dict_shade.items():
        cl = 'None'
        if val:
            cl = 'k'
        temp_elements.append(Line2D([0], [0], marker='h', color='k', label='\SI{%d}{\celsius}'%key, ls = 'None', \
             markerfacecolor=cl, markeredgewidth = 2, markersize=msize))
    
    ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
    fs = 9
    bta = [(1, 0.25), (0.01, 1), (0.01, 0.73)]
    
    l1 = ax.legend(title = 'Strain Range', 
              handles=strain_elements,
              loc='upper left',
              bbox_to_anchor=bta[1],
              edgecolor = 'k',
              facecolor = 'w',
              framealpha = 1,
              fontsize = fs)
    
    ax.add_artist(l1)
    
    ax.tick_params(axis = 'both', direction='in', which = 'both')
    
    if save:
        plt.savefig(os.path.join(path, save))
    
    plt.show()
    
def graph_test_train_hists(data, bin_width = 10, load = True, save = '', ax = None):
    
    if load:
        print(data)
        d = np.load(data)
    else:
        d = data
        
    y_obs1, y_pred1 = d['y_obs_test'].reshape(11,-1), d['y_pred_test'].reshape(11,-1)
    y_obs0, y_pred0 = d['y_obs_train'].reshape(33,-1), d['y_pred_train'].reshape(33,-1)

    ex_ax = False
    if ax:
        ex_ax = True
    else:
        fig, ax = plt.subplots(figsize=(4,4))
    
    y_diff0 = (y_pred0 - y_obs0)/y_obs0*100
    y_diff0 = y_diff0.flatten()
    
    y_diff1 = (y_pred1 - y_obs1)/y_obs1*100
    y_diff1 = y_diff1.flatten()
    
    bins = range(-200, 200+bin_width, bin_width)
    
    ax.hist(y_diff1, bins = bins, color = '#00b300', ec="k", alpha = 0.4, weights=np.ones(len(y_diff1))/len(y_diff1), label = 'Train')
    ax.hist(y_diff0, bins = bins, color = '#8000ff', ec="k", alpha = 0.4, weights=np.ones(len(y_diff0))/len(y_diff0), label = 'Test')
    
    ax.set_xlim(-200, 200)
    ax.set_ylim(0, 0.4)
    # if max(abs(y_diff1)) > 100:
    #     ax.set_xlim(-200, 200)
    # if max(abs(y_diff1)) > 200:
    #     ax.set_xlim(-300, 300)
    
    ax.set_xlabel('Percentage Error (\%)')
    ax.set_ylabel('Frequency')
    
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_ticks(np.arange(0, 0.5, 0.1))
    
    ax.legend(loc='upper left', edgecolor = 'k', framealpha = 1)
    
    if save:
        plt.savefig(os.path.join(path, save), bbox_inches = 'tight')
        
    if not ex_ax:
        plt.show()
    
    return max(abs(y_diff0)), max(abs(y_diff1))