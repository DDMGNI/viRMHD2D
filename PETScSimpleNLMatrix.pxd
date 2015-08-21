'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py.PETSc cimport DMDA, Mat, Vec

from PETScDerivatives cimport PETScDerivatives


cdef class PETScMatrix(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    cdef DMDA da1
    cdef DMDA da4
    
    cdef Vec Xh
    
    cdef Vec localB
    cdef Vec localX
    cdef Vec localXh
    
    cdef PETScDerivatives derivatives
