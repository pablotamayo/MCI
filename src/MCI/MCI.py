# MCC entropy

import pandas as pd
import numpy as np
from .compute_ranks_and_copula_dens_multigrid_v2 import compute_ranks_and_copula_dens_multigrid_v2
from .compute_copula_entropies import compute_copula_entropies
from sys import float_info

def MCI(x, y, binary_x = False, binary_y = False, rseed = 8672439): 
    # Multiscale Symmetrized Cumulative Copula Entropy 

    if isinstance(x, pd.Series):
        x = x.to_numpy()
        
    x = x.astype(float)
        
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    y = y.astype(float)

    n = len(x)
    
    if binary_x:
        np.random.seed(rseed)
        noise = np.random.normal(0, float_info.epsilon, n)
        x += noise

    if binary_y:
        np.random.seed(rseed)
        noise= np.random.normal(0, float_info.epsilon, n)
        y += noise

    grid_4, grid_8, grid_16, grid_32  = compute_ranks_and_copula_dens_multigrid_v2(x, y, n)    
    
    grid_4 += np.finfo(np.float64).tiny 
    grid_8 += np.finfo(np.float64).tiny 
    grid_16 += np.finfo(np.float64).tiny 
    grid_32 += np.finfo(np.float64).tiny 

    S_p_p_4, S_m_p_4, S_p_m_4, S_m_m_4 = compute_copula_entropies(grid_4, 4)
    S_p_p_8, S_m_p_8, S_p_m_8, S_m_m_8 = compute_copula_entropies(grid_8, 8)
    S_p_p_16, S_m_p_16, S_p_m_16, S_m_m_16 = compute_copula_entropies(grid_16, 16)
    S_p_p_32, S_m_p_32, S_p_m_32, S_m_m_32 = compute_copula_entropies(grid_32, 32)
    
    SCCE_4 = S_p_p_4 + S_m_m_4 - S_p_m_4 - S_m_p_4 
    SCCE_8 = S_p_p_8 + S_m_m_8 - S_p_m_8 - S_m_p_8
    SCCE_16 = S_p_p_16 + S_m_m_16 - S_p_m_16 - S_m_p_16
    SCCE_32 = S_p_p_32 + S_m_m_32 - S_p_m_32 - S_m_p_32
    
    SCCE_array = np.array([SCCE_4, SCCE_8, SCCE_16, SCCE_32]) 
    SCCE_abs_array = np.abs(SCCE_array)
    SCCE_argmax = np.argmax(SCCE_abs_array)
    SCCE_max = SCCE_array[SCCE_argmax]

    SCCE_sign = np.sign(SCCE_max)
    SCCE_val = np.abs(SCCE_max)
    
    #a = 2.99319243  # values to fit n = 5000
    #b =  2.70587669
    #c = 0.28656317
    #d = -3.27931520
    a =  2.97579373   # values to fit n = 1000
    b = 2.81571180 
    c = 0.18098956 
    d = -3.15693695 
    
    SCCE_norm = SCCE_sign * (a + b*SCCE_val + c*np.exp(SCCE_val) + d*np.exp(SCCE_val**3))
    if SCCE_norm > 1.0:
        SCCE_norm = 1.0
    if SCCE_norm < - 1.0:
        SCCE_norm = - 1.0        
    
    return (np.round(SCCE_norm, 5).item())

