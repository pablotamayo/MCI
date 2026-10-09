cimport cython
cimport numpy as np
import numpy as pynp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)

cpdef compute_ranks_and_copula_dens_multigrid_v2(
                                              np.ndarray[np.float_t] x, 
                                              np.ndarray[np.float_t] y,   
                                              int n):
    
    cdef int i, a_32, b_32
    cdef np.float grid_scaling_factor, one_over_n
    
    cdef np.ndarray[np.float_t, ndim = 2] grid_4 = pynp.zeros((4, 4), dtype = float) 
    cdef np.ndarray[np.float_t, ndim = 2] grid_8 = pynp.zeros((8, 8), dtype = float) 
    cdef np.ndarray[np.float_t, ndim = 2] grid_16 = pynp.zeros((16, 16), dtype = float) 
    cdef np.ndarray[np.float_t, ndim = 2] grid_32 = pynp.zeros((32, 32), dtype = float)   
    cdef np.ndarray[np.int64_t, ndim = 1] seq2 = pynp.empty((n), dtype = int)    
    cdef np.ndarray[np.int64_t, ndim = 1] x_ranks = pynp.empty((n), dtype = int)
    cdef np.ndarray[np.int64_t, ndim = 1] y_ranks = pynp.empty((n), dtype = int)
    cdef np.ndarray[np.int64_t] x_order 
    cdef np.ndarray[np.int64_t] y_order 
     
    grid_scaling_factor = 31.999999/(float(n) - 1.0)
    one_over_n = 1.0/float(n)
          
    seq2 = (pynp.arange(float(n)) * grid_scaling_factor).astype(int)
    x_order = x.argsort()
    y_order = y.argsort()
    
    x_ranks[x_order] = seq2
    y_ranks[y_order] = seq2

    for i in range(n):
        a_32 = x_ranks[i]
        b_32 = y_ranks[i]
        grid_32[a_32, b_32] += one_over_n
        grid_16[a_32//2, b_32//2] += one_over_n  
        grid_8[a_32//4, b_32//4] += one_over_n       
        grid_4[a_32//8, b_32//8] += one_over_n        
        
    return([grid_4, grid_8, grid_16, grid_32])