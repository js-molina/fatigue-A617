import pandas as pd
import numpy as np
from ..finder.cycle_path import cycle_path_from_test, peak_path_from_test

def get_cycles_from_test(test):
    path = cycle_path_from_test(test)

    df = pd.read_csv(path)

    cycles = []
    for c in df['Cycle Label'].unique():
        cycles.append(df.loc[df['Cycle Label'] == c])
    
    return cycles

def get_peak_data_from_test(test):

    path = peak_path_from_test(test)

    df = pd.read_csv(path)
    
    # t1 = pd.to_numeric(df['Min Stress Mpa'], errors='coerce').isna().sum()
    # t2 = pd.to_numeric(df['Max Stress Mpa'], errors='coerce').isna().sum()
    
    # print(t1+t2, end = ', ')

    # df = df[pd.to_numeric(df['Min Stress Mpa'], errors='coerce').notnull()]
    # df = df[pd.to_numeric(df['Max Stress Mpa'], errors='coerce').notnull()]
    
    df['Min Stress Mpa'] = pd.to_numeric(df['Min Stress Mpa'], errors='coerce').interpolate()
    df['Max Stress Mpa'] = pd.to_numeric(df['Max Stress Mpa'], errors='coerce').interpolate()
    
    clean_df = df.copy()

    clean_df['Max Stress Mpa'] = pd.to_numeric(df['Max Stress Mpa'])
    clean_df['Min Stress Mpa'] = pd.to_numeric(df['Min Stress Mpa'])
    clean_df['Stress Ratio'] = df['Max Stress Mpa']/(-df['Min Stress Mpa'])

    return clean_df


def chi_ratio(pred, obs):
    return np.sum((pred/obs-1)**2, axis = 0).mean()  

def meape(pred, obs):
    return np.mean(abs((pred-obs)/obs)*100)