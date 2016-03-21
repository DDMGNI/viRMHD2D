'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

from petsc4py.PETSc cimport Mat, SNES, Vec

from PETScDerivatives cimport PETScDerivatives


cdef class PETScSolverDOF2(object):

    cdef int  nx
    cdef int  ny
    
    cdef double ht
    cdef double hx
    cdef double hy
    
    cdef double ht_inv
    cdef double hx_inv
    cdef double hy_inv
    
    cdef object da1
    cdef object da2
    
    cdef object pc
    
    cdef Vec Yd
    
    cdef Vec FA
    cdef Vec FJ
    cdef Vec FP
    cdef Vec FO
    
    cdef public Vec Ap
    cdef public Vec Jp
    cdef public Vec Pp
    cdef public Vec Op
    
    cdef public Vec Ah
    cdef public Vec Jh
    cdef public Vec Ph
    cdef public Vec Oh
    
    cdef public Vec Aa
    cdef public Vec Ja
    cdef public Vec Pa
    cdef public Vec Oa
    
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
