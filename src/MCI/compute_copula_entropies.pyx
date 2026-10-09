cimport cython
cimport numpy as np
import numpy as pynp
from libc.math cimport log2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)

cpdef compute_copula_entropies(np.ndarray[np.float_t, ndim=2] c_xy, int g):
    
    cdef int i, j
    cdef np.float_t s, s_p, s_m, S_p_p, S_p_m, S_m_p, S_m_m

    cdef np.ndarray[np.float_t, ndim = 2] C_p_dot = pynp.empty((g, g), dtype = float)    
    cdef np.ndarray[np.float_t, ndim = 2] C_m_dot = pynp.empty((g, g), dtype = float)    
     
    for j in range(g):
        s_p = 0
        s_m = 0
        for i in range(g):
            s_p += c_xy[i, j]
            C_p_dot[i, j] = s_p
            
            s_m += c_xy[g - 1 - i, j]
            C_m_dot[g - 1 - i, j] = s_m
            
    S_p_p = 0
    S_p_m = 0
    for i in range(g):
        s_p = 0
        s_m = 0
        for j in range(g):
            s_p += C_p_dot[i, j]
            S_p_p += s_p * log2(s_p)
            
            s_m += C_p_dot[i, g - 1 - j]
            S_p_m += s_m * log2(s_m)           
            
    S_m_p = 0
    S_m_m = 0
    for i in range(g):
        s_p = 0
        s_m = 0
        for j in range(g):
            s_p += C_m_dot[i, j]      
            S_m_p += s_p * log2(s_p)
            
            s_m += C_m_dot[i, g - 1 - j] 
            S_m_m += s_m * log2(s_m)
            
    unit_area = 1.0/(g*g)        
    S_p_p = - unit_area * S_p_p
    S_m_p = - unit_area * S_m_p
    S_p_m = - unit_area * S_p_m
    S_m_m = - unit_area * S_m_m
            
    return([S_p_p, S_m_p, S_p_m, S_m_m])    