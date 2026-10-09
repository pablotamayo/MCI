# MCI entropy
import pandas as pd
import numpy as np
from .MCI import MCI

def MCI_assoc(x, y, binary_x = False, binary_y = False, rseed = 8672439):
    # Multiscale Symmetrized Cumulative Copula Entropy for multiple x's and y's
      
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()

    if isinstance(y, pd.DataFrame):
        y = y.to_numpy()        
        
    x = x.astype(float)
    y = y.astype(float)
    
    if x.ndim == 1 and y.ndim == 1:        
        return(MCI(x, y, binary_x = binary_x, binary_y = binary_y, rseed = rseed))
    
    if x.ndim == 2 and y.ndim == 2:
        scores = []  
        for i in range(x.shape[0]):     
            scores.append(MCI(x[i, :], y[i, :], binary_x = binary_x, binary_y = binary_y, rseed = rseed))
        return scores
    
    if x.ndim == 2 and y.ndim == 1:
        scores = []
        for i in range(x.shape[0]):
            scores.append(MCI(x[i, :], y, binary_x = binary_x, binary_y = binary_y, rseed = rseed))
        return(scores)    
        
    if x.ndim == 1 and y.ndim == 2:
        scores = []
        for i in range(y.shape[0]):
            scores.append(MCI(x, y[i, :], binary_x = binary_x, binary_y = binary_y, rseed = rseed))
        return(scores)
