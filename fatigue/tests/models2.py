import matplotlib.pyplot as plt

from ..graph.models2 import graph_model, graph_prediction
from ..models2 import *

def test_morrow2(data):
    model = morrow2(data)
    graph_model(model)
    graph_prediction(model)
    
    model = plastic_manson(data)
    graph_model(model)
    graph_prediction(model)
    
    model = coffin_manson(data)
    graph_model(model)
    graph_prediction(model)
    

def test_empirical(data):
    model = morrow2(data)
    graph_prediction(model)
    
    model = plastic_manson(data)
    graph_prediction(model)
    
    model = coffin_manson(data)
    graph_prediction(model)
    