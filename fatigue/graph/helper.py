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

    df = df[pd.to_numeric(df['Min Stress Mpa'], errors='coerce').notnull()]
    df = df[pd.to_numeric(df['Max Stress Mpa'], errors='coerce').notnull()]
    
    clean_df = df.copy()

    clean_df['Max Stress Mpa'] = pd.to_numeric(df['Max Stress Mpa'])
    clean_df['Min Stress Mpa'] = pd.to_numeric(df['Min Stress Mpa'])
    clean_df['Stress Ratio'] = pd.to_numeric(df['Stress Ratio'])

    return clean_df


def chi_ratio(pred, obs):
    return np.sum((pred/obs-1)**2, axis = 0).mean()   
