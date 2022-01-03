#%%
import random
from fatigue.finder import fatigue_data
from fatigue.finder import cycle_path
from fatigue.tests.properties import test_plastic_strain, test_strain_from_cycles
from fatigue.tests.models import test_morrow
from fatigue.tests.strain import test_strain_vals
from fatigue.tests.models2 import test_morrow2
from fatigue.tests.peaks import *
import fatigue.graph as gr
import fatigue.strain as st
from fatigue.filter import test_filter
from fatigue.networks import vectorise_data

# print('Naive')
# test_morrow(fatigue_data)6
# print('Normalised')
test_morrow2(fatigue_data)
# test_strain_vals(fatigue_data)

# test = fatigue_data.data[6]

# test = random.choice(fatigue_data.data)
# test, = fatigue_data.get_test_from_sample('J3')

# # test_plastic_strain(test)

# test_strain_from_cycles(test)
# print()
# test_strain_from_peaks(test, [1, 2, 5, 10, 20, 50, 99, 199, 349])

# test_scuffed_energy(test, [1, 2, 5, 10, 20, 50, 99, 199, 349])

# gr.graph_peaks_from_test(test)

# test_filter(fatigue_data.get_data(850)[-1], lowest_cov = True)

# for test in fatigue_data.data:
#     gr.graph_peaks_from_test(test)

# r = test_some_data(test)

# X, y = vectorise_data(fatigue_data.data)


# X = test_features(fatigue_data.data)
# 


#%% 

gr.models2.graph_nn_prediction('ydata.npz')


