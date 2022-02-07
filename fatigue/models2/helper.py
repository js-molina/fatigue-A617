from ..finder.cycle_path import peak_path_from_test, peak_path_from_sample
import pandas as pd

TEMPS = [850, 950]

def get_nf(test, from_sample = False):
    if from_sample:
        path = peak_path_from_sample(test)
    else:
        path = peak_path_from_test(test)
    df = pd.read_csv(path)
    return len(df.Cycle)