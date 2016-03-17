'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py.PETSc cimport Mat, SNES, Vec

from PETScDerivatives cimport PETScDerivatives


cdef class PETScVorticity(object):

    cdef int  nx
    cdef int  ny
    
    cdef double ht
    cdef double hx
    cdef double hy
    
    cdef double ht_inv
    cdef double hx_inv
    cdef double hy_inv
    
    cdef double arakawa_fac
    
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
