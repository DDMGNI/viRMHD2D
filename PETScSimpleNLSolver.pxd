'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py.PETSc cimport DMDA, Mat, SNES, Vec

from PETScDerivatives cimport PETScDerivatives


cdef class PETScSolver(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    cdef np.float64_t ht_inv
    cdef np.float64_t hx_inv
    cdef np.float64_t hy_inv
    
    cdef DMDA da1
    cdef DMDA da4
    
    cdef Vec Xh
    cdef Vec Xp
    
    cdef Vec localXp
    cdef Vec localXh
    
    cdef PETScDerivatives derivatives
