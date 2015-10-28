'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py.PETSc cimport Mat, SNES, Vec

from PETScDerivatives cimport PETScDerivatives


cdef class PETScOhmsLaw(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    cdef np.float64_t ht_inv
    cdef np.float64_t hx_inv
    cdef np.float64_t hy_inv
    
    cdef object da1
    
    cdef Vec Ah
    cdef Vec Ph
    
    cdef Vec localAp
    cdef Vec localAh
    cdef Vec localPh
    
    cdef PETScDerivatives derivatives
