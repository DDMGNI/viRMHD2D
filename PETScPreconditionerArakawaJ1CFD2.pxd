'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

from petsc4py.PETSc cimport KSP, Mat, SNES, Vec

from PETScDerivatives cimport PETScDerivatives
from PETScPoissonCFD2 cimport PETScPoisson


cdef class PETScPreconditioner(object):

    cdef int  nx
    cdef int  ny
    
    cdef double ht
    cdef double hx
    cdef double hy
    
    cdef double ht_inv
    cdef double hx_inv
    cdef double hy_inv
    
    cdef double arakawa_fac
    cdef double arakawa_fac2
    
    cdef int jacobi_max_it
    
    cdef object da1
    cdef object da4
    
    cdef Vec Ad
    cdef Vec Jd
    cdef Vec Pd
    cdef Vec Od
    
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
    
    cdef Vec F
    cdef Vec L
    cdef Vec T
    
    cdef Vec FA
    cdef Vec FJ
    cdef Vec FP
    cdef Vec FO
    
    cdef Vec Pb
    cdef Mat Pm
    cdef Vec Qb
    cdef Mat Qm

    cdef Vec localL
    cdef Vec localQ
    cdef Vec localT

    cdef Vec localFA
    cdef Vec localFJ
    cdef Vec localFP
    cdef Vec localFO
    
    cdef Vec localAa
    cdef Vec localPa
    
    cdef Vec localAd
    cdef Vec localPd
    
    cdef PETScDerivatives derivatives
    cdef PETScPoisson     petsc_poisson
    
    cdef object poisson_nullspace
    
    cdef KSP poisson_ksp
    cdef KSP parabol_ksp
    
    cdef double[:,:] aa
    cdef double[:,:] pa
    
    cdef double[:,:] fa
    cdef double[:,:] fj
    cdef double[:,:] fp
    cdef double[:,:] fo
    
    