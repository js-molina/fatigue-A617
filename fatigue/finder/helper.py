import os

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