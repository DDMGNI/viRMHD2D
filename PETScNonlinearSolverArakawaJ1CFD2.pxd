'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py.PETSc cimport Mat, SNES, Vec

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
    
    cdef object da1
    cdef object da4
    
    cdef object pc
    
    cdef Vec Xh
    cdef Vec Xp
    cdef Vec Yd
    
    cdef Vec Ap
    cdef Vec Jp
    cdef Vec Pp
    cdef Vec Op
    
    cdef Vec Ah
    cdef Vec Jh
    cdef Vec Ph
    cdef Vec Oh
    
    cdef Vec Aa
    cdef Vec Ja
    cdef Vec Pa
    cdef Vec Oa
    
    cdef Vec YA
    cdef Vec YJ
    cdef Vec YP
    cdef Vec YO
    
    cdef Vec Ad
    cdef Vec Jd
    cdef Vec Pd
    cdef Vec Od
    
    cdef Vec T
    cdef Vec T1
    cdef Vec T2
    cdef Vec T3
    cdef Vec T4
    
    cdef Vec localAa
    cdef Vec localJa
    cdef Vec localPa
    cdef Vec localOa
    
    cdef PETScDerivatives derivatives
