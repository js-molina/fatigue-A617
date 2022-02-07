import os
import pandas as pd
import numpy as np

def get_trs_from_path(folder, pager = {}):
    ss = folder.split(os.sep)
    temp = int(ss[-3][:-1])
    rate = float(ss[-2].split()[-1])
    strain = float(ss[-1].split()[0][:-1])
    pager[(temp, 'T')] = ss[-3]
    pager[(rate, 'R')] = ss[-2]
    pager[(strain, 'S')] = ss[-1]
    return temp, rate, strain

def get_ss_from_file(fileName, pager):
    sample = fileName.split('.')[0]
    pager[(sample, 'S2')] = fileName
    return sample

def fd_to_df(data):
    temp, rate, strain, sample = [], [] ,[] ,[]
    for test in data:
        temp.append(test.Temp)
        rate.append(test.Rate)
        strain.append(test.Strain)
        sample.append(test.Sample)
    return pd.DataFrame(zip(temp, rate, strain, sample), columns = ['Temps', 'Rates', 'Strains', 'Samples'])
        