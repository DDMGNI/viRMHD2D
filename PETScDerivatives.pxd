'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

from petsc4py.PETSc cimport Vec


cdef class PETScDerivatives(object):
    '''
    Cython Implementation of MHD Discretisation
    '''
    
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
    
    cdef Vec localX
    cdef Vec localY
    
    
    cdef double arakawa(self, double[:,:] x, double[:,:] h, int i, int j)
    cdef double laplace(self, double[:,:] x, int i, int j)
    
    cpdef arakawa_vec(self, Vec X, Vec Y, Vec A)
    cpdef laplace_vec(self, Vec X, Vec Y, double sign)
    
    cpdef double dx(self, Vec X, Vec D, double sign)
    cpdef double dy(self, Vec X, Vec D, double sign)
