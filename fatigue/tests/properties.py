import matplotlib.pyplot as plt

from ..graph import get_cycles_from_test, graph_cycle, graph_filtered_cycle
from ..strain import *

def test_plastic_strain(trial):
    
    cycles = get_cycles_from_test(trial)

    for cycle in cycles:
        ax = plt.gca()
        ax.set_title(cycle['Cycle Label'].iloc[0])
        graph_cycle(cycle, 'gray', ax)
        # graph_filtered_cycle(cycle, ax, 'blue')
        x = get_plastic_zeros(cycle)
        ax.plot(x, [0, 0], 'ro', markersize = 5)
        ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)
    
        plt.show()
    
    print('Plastic Strain = %.2e'%total_plastic_strain_percent(trial))


def test_elastic_strain_from_cycles(trial):
    cycles = get_cycles_from_test(trial)

    for cycle in cycles:
        print(cycle['Cycle Label'].iloc[0])
        print('Elastic Strain = %.2f'%cycle_elastic_strain_percent(cycle, trial.Temp))
        
def test_strain_from_cycles(trial):
    cycles = get_cycles_from_test(trial)

    for cycle in cycles:
        print(cycle['Cycle Label'].iloc[0])
        print('Elastic Strain = %.2f'%cycle_elastic_strain_percent(cycle, trial.Temp))
        print('Plastic Strain = %.2f'%cycle_plastic_strain_percent(cycle))

def test_plastic_energy(trial):    
    print('Plastic Strain = %.2e'%total_plastic_strain(trial))
    

