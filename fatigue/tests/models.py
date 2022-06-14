import matplotlib.pyplot as plt

from ..graph import graph_model, graph_prediction
from ..models import *

def test_morrow(data):
    model = morrow(data)
    # graph_model(model, save='mod1.svg')
    graph_prediction(model, save='pred1.pdf')
    
    model = plastic_manson(data)
    # graph_model(model, save='mod2.svg')
    graph_prediction(model, save='pred2.pdf')
    
    model = coffin_manson(data)
    # graph_model(model, save='mod3.svg')
    graph_prediction(model, save='pred3.pdf')
    
    model = goswami_m(data)
    # graph_model(model)
    graph_prediction(model, save='pred4.pdf')

    

