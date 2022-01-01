import matplotlib.pyplot as plt

from ..graph import graph_model, graph_prediction
from ..models import *

def test_morrow(data):
    model = morrow(data)
    graph_model(model)
    graph_prediction(model)
    
    model = plastic_manson(data)
    graph_model(model)
    graph_prediction(model)
    
    model = coffin_manson(data)
    graph_model(model)
    graph_prediction(model)
    


    

