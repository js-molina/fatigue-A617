#%%
import os
from . import fatigue_data, path, path_dict

def peak_path_from_test(data, pager = path_dict):
    new_path = os.sep.join([path, pager[(data.Temp, 'T')], pager[(data.Rate, 'R')], \
                          pager[(data.Strain, 'S')], pager[(data.Sample, 'S1')]])
    return new_path + 'PeakStress_' + pager[(data.Sample, 'S2')]
    
def cycle_path_from_test(data, pager = path_dict):
    new_path = os.sep.join([path, pager[(data.Temp, 'T')], pager[(data.Rate, 'R')], \
                          pager[(data.Strain, 'S')], pager[(data.Sample, 'S1')]])
    return new_path + 'Cycles_' + pager[(data.Sample, 'S2')]

def peak_path_from_sample(sample, pager = path_dict):
    data = [data for data in fatigue_data.data if data.Sample == sample][0]
    return(peak_path_from_test(data, pager))

def cycle_path_from_sample(sample, pager = path_dict):
    data = [data for data in fatigue_data.data if data.Sample == sample][0]
    return(cycle_path_from_test(data, pager))

