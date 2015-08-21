'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

from petsc4py.PETSc cimport DMDA, Vec


cdef class PETScDerivatives(object):
    '''
    Cython Implementation of MHD Discretisation
    '''
    
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
    
    cdef Vec localX
    
    cdef np.ndarray ty
    
    
    cdef double arakawa(self, double[:,:] x, double[:,:] h,
                              np.uint64_t i, np.uint64_t j)

    cdef double laplace(self, double[:,:] x,
                              np.uint64_t i, np.uint64_t j)
    
    cpdef laplace_vec(self, Vec X, Vec D, np.float64_t sign)
    
    cpdef double dx(self, Vec X, Vec D, np.float64_t sign)
    
    cpdef double dy(self, Vec X, Vec D, np.float64_t sign)

    cdef  double dt(self, double[:,:] x, np.uint64_t i, np.uint64_t j)

