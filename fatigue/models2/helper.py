from ..finder.cycle_path import peak_path_from_test
import pandas as pd

TEMPS = [850, 950]

def get_nf(test):
    path = peak_path_from_test(test)
    df = pd.read_csv(path)
    return len(df.Cycle)