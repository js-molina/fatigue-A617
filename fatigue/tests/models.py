import matplotlib.pyplot as plt

from ..graph import graph_model, graph_prediction
from ..models import *

def test_morrow(data):
    model = morrow(data)
    graph_model(model, save='mod1.svg')
    graph_prediction(model, save='pred1.svg')
    
    model = plastic_manson(data)
    graph_model(model, save='mod2.svg')
    graph_prediction(model, save='pred2.svg')
    
    model = coffin_manson(data)
    graph_model(model, save='mod3.svg')
    graph_prediction(model, save='pred3.svg')
    


    

