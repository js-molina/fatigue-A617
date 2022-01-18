import matplotlib.pyplot as plt
import os

from ..graph.models2 import graph_model, graph_prediction
from ..models2 import *

def test_morrow2(data):
    
    path = r'D:\WSL\ansto\figs'
    
    model = morrow2(data)
    graph_model(model, os.path.join(path, 'morrow-m.pdf'))
    graph_prediction(model, os.path.join(path, 'morrow-p.pdf'))
    
    model = plastic_manson(data)
    graph_model(model, os.path.join(path, 'pmanson-m.pdf'))
    graph_prediction(model, os.path.join(path, 'pmanson-p.pdf'))
    
    model = coffin_manson(data)
    graph_model(model, os.path.join(path, 'cmanson-m.pdf'))
    graph_prediction(model, os.path.join(path, 'cmanson-p.pdf'))
    

def test_empirical(data):
    model = morrow2(data)
    graph_prediction(model)
    
    model = plastic_manson(data)
    graph_prediction(model)
    
    model = coffin_manson(data)
    graph_prediction(model)
    