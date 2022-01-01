from .helper import *
from .morrow2 import *
from .pl_manson2 import *
from .c_manson2 import *

class model:
    def __init__(self, model_data, ntype):
        self.values, self.fun, self.lc = model_data
        self.model_type = ntype
    
    def _get_plot_params(self):
        return self.values, self.fun, self.lc
    
    def _get_model_type(self):
        return self.model_type

class morrow2(model):
    def __init__(self, fatigue_data):
        model.__init__(self, morrow_construct2(fatigue_data), 'morrow')
        self.pred = []
        for li in self.lc:
            self.pred.append(lambda x, li = li : morrow_eqn2(x, *li))
        
class plastic_manson(model):
    def __init__(self, fatigue_data):
        model.__init__(self, plmanson_construct2(fatigue_data), 'pl_manson')
        self.pred = []
        for li in self.lc:
            self.pred.append(lambda x, li = li : plmanson_eqn2(x, *li))

class coffin_manson(model):
    def __init__(self, fatigue_data):
        model.__init__(self, cmanson_construct2(fatigue_data), 'c_manson')
        self.pred = []
        for li in self.lc:
            self.pred.append(lambda x, li = li : cmanson_eqn2(x, *li))

              