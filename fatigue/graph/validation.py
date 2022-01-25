import matplotlib.pyplot as plt
import numpy as np

from .helper import chi_ratio

def plot_history_loss(history, title = 'L1/L2 Activity Loss'):

    ax = plt.gca()
    
    ax.plot(history.history['loss'], 'b-', lw = 0.9, label='Training data')
    ax.plot(history.history['val_loss'], 'r-', lw = 0.9, label='Validation data')
    ax.set_title(title)
    ax.set_ylabel('Loss value')
    ax.set_xlabel('No. epoch')
    
    ax.legend(framealpha = 1, edgecolor = 'None')
    
    plt.show()
    
def plot_history_mape(history, title = 'L1/L2 Mean Absolute Percentage Error'):

    ax = plt.gca()
    
    ax.plot(history.history['mean_absolute_percentage_error'], 'b-', lw = 0.9, label='Training data')
    ax.plot(history.history['mean_absolute_percentage_error'], 'r-', lw = 0.9, label='Validation data')
    ax.set_title(title)
    ax.set_ylabel('Mean Absolute Percentage Error (\%)')
    ax.set_xlabel('No. epoch')
    
    ax.legend(framealpha = 1, edgecolor = 'None')
    
    plt.show()