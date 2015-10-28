'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py.PETSc cimport Mat, SNES, Vec

from PETScDerivatives cimport PETScDerivatives


cdef class PETScVorticity(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    cdef np.float64_t ht_inv
    cdef np.float64_t hx_inv
    cdef np.float64_t hy_inv
    
    cdef object da1
    
    cdef Vec Oh
    cdef Vec Pp
    cdef Vec Ph
    cdef Vec Ah
    cdef Vec Jh
    
    cdef Vec localOp
    cdef Vec localOh
    cdef Vec localPp
    cdef Vec localPh
    cdef Vec localAh
    cdef Vec localJh
    
    cdef PETScDerivatives derivatives
