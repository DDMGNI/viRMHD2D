'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py.PETSc cimport Vec


cdef class PETScPoisson(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    cdef object da1
    
    cdef Vec localB
    cdef Vec localX
