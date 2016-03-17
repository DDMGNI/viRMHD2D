'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py.PETSc cimport Mat, SNES, Vec

from PETScDerivatives cimport PETScDerivatives


cdef class PETScOhmsLaw(object):

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
    
    cdef Vec Ah
    cdef Vec Ph
    
    cdef Vec localAp
    cdef Vec localAh
    cdef Vec localPh
    
    cdef PETScDerivatives derivatives
