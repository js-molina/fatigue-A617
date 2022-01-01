#%%
from .helper import *
import os, collections

main_path = os.path.abspath(__file__ + "/../../../")

path = os.path.join(main_path, 'data')

Test = collections.namedtuple('Test', ['Temp', 'Rate', 'Strain', 'Sample'])

def populate_from_path(path):
    datum = []
    path_names = {}
    for folderName, subFolders, fileNames in os.walk(path):
        if folderName.endswith('strain'):
            temp, rate, strain = get_trs_from_path(folderName, path_names)
        samples = set([filename.split('_')[-1] for filename in fileNames if filename.endswith('.csv')])
        for filename in samples:
            if filename.endswith('.csv'):
                sample = get_ss_from_file(filename, path_names)
                for file in fileNames:
                    if file.endswith(filename):
                        if 'Cycle' in file:
                            path_names[(sample, 'S1')] = file.split('Cycle')[0]
                datum.append(Test(temp, rate, strain, sample))
    return datum, path_names

class Data:
    def __init__(self, datum):
        self.data = datum
    
    def get_data(self, *char, samples = False):
        tmp = []
        for el in self.data:
            if el.Temp == char[0]:
                if len(char) > 1:
                    if el.Rate == char[1]:
                        if len(char) > 2:
                            if el.Strain == char[2]:
                                tmp.append(el)
                                continue
                            else:
                                continue
                        tmp.append(el)
                        continue
                    else:
                        continue
                tmp.append(el)
        if samples == True:
            sam = []
            for el in tmp:
                sam.append(el.Sample)
            return sam
        else:
            return tmp
    
    def get_test_from_sample(self, sample):
        return [i for i in fatigue_data.data if i.Sample == sample]

tests, path_dict = populate_from_path(path)
fatigue_data = Data(tests)

if __name__ == "__main__" and __package__ is None:
    __package__ = "finder"