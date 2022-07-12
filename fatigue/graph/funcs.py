import matplotlib.pyplot as plt
from ..strain.helper import filter_by_savgol, RATIO

def graph_cycle(c, color = 'gray', ax = None, flush = False):
    strain = c.Strain
    stress = c['Stress Mpa']

    if not ax:
        ax = plt.gca()
        ax.set_xlabel("Strain (mm/mm)")
        ax.set_ylabel("Stress (MPa)")
        ax.set_title(c['Cycle'].iloc[0])
        
    ax.plot(strain, stress, lw = 0.8, color = color)
    
    if flush:
        plt.show()
    else:
        return ax
    

def graph_filtered_cycle(c, color = 'gray', ax = None, flush = False):
    
    strain = c.Strain
    stress = filter_by_savgol(c['Stress Mpa'], RATIO, 2)

    if not ax:
        ax = plt.gca()
        ax.set_xlabel("Strain (mm/mm)")
        ax.set_ylabel("Stress (MPa)")
    
    ax.plot(strain, stress, lw = 0.8, color = color)
    
    if flush:
        plt.show()
    else:
        return ax
    
    
    
    