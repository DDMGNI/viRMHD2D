'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py.PETSc cimport Vec


cdef class PETScPoisson(object):

    cdef int  nx
    cdef int  ny
    
    cdef double hx
    cdef double hy
    
    cdef double lapx_fac
    cdef double lapy_fac
    
    cdef object da1
    
    cdef Vec localX
