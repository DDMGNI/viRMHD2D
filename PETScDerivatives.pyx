'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport DMDA, Vec


cdef class PETScDerivatives(object):
    '''
    Cython Implementation of MHD Discretisation
    '''
    
    
    def __cinit__(self, DMDA da1, DMDA da4,
                  np.uint64_t  nx, np.uint64_t  ny,
                  np.float64_t ht, np.float64_t hx, np.float64_t hy):
        '''
        Constructor
        '''
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        self.ht_inv = 1. / ht
        self.hx_inv = 1. / hx
        self.hy_inv = 1. / hy
        
        
        # distributed arrays
        self.da1 = da1
        self.da4 = da4
        
        
        # create local vectors
        self.localX = da1.createLocalVec()
        
        
    
    @cython.boundscheck(False)
    cdef np.float64_t arakawa(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.ndarray[np.float64_t, ndim=2] h,
                                    np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: Arakawa Bracket
        '''
        
        cdef np.float64_t jpp, jpc, jcp, result
        
        jpp = (x[i+1, j  ] - x[i-1, j  ]) * (h[i,   j+1] - h[i,   j-1]) \
            - (x[i,   j+1] - x[i,   j-1]) * (h[i+1, j  ] - h[i-1, j  ])
        
        jpc = x[i+1, j  ] * (h[i+1, j+1] - h[i+1, j-1]) \
            - x[i-1, j  ] * (h[i-1, j+1] - h[i-1, j-1]) \
            - x[i,   j+1] * (h[i+1, j+1] - h[i-1, j+1]) \
            + x[i,   j-1] * (h[i+1, j-1] - h[i-1, j-1])
        
        jcp = x[i+1, j+1] * (h[i,   j+1] - h[i+1, j  ]) \
            - x[i-1, j-1] * (h[i-1, j  ] - h[i,   j-1]) \
            - x[i-1, j+1] * (h[i,   j+1] - h[i-1, j  ]) \
            + x[i+1, j-1] * (h[i+1, j  ] - h[i,   j-1])
        
        result = (jpp + jpc + jcp) / (12. * self.hx * self.hy)
        
        return result
    
    
    @cython.boundscheck(False)
    cdef np.float64_t arakawa_grad(self, np.ndarray[np.float64_t, ndim=2] x,
                                         np.ndarray[np.float64_t, ndim=2] h,
                                         np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: Arakawa Bracket
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 0.5 * x[i-2, j-1] * h[i-1, j-1] \
                     + 0.5 * x[i-2, j-1] * h[i-1, j+1] \
                     - 1.  * x[i-2, j-1] * h[i-1, j  ] \
                     - 0.5 * x[i-2, j+1] * h[i-1, j-1] \
                     - 0.5 * x[i-2, j+1] * h[i-1, j+1] \
                     + 1.  * x[i-2, j+1] * h[i-1, j  ] \
                     + 1.  * x[i-1, j-2] * h[i,   j-2] \
                     - 1.  * x[i-1, j-2] * h[i,   j-1] \
                     - 0.5 * x[i-1, j-1] * h[i-2, j-1] \
                     + 0.5 * x[i-1, j-1] * h[i-2, j+1] \
                     + 4.  * x[i-1, j-1] * h[i-1, j+1] \
                     - 4.  * x[i-1, j-1] * h[i-1, j  ] \
                     + 0.5 * x[i-1, j-1] * h[i,   j-1] \
                     - 0.5 * x[i-1, j-1] * h[i,   j+1] \
                     - 0.5 * x[i-1, j+1] * h[i-2, j-1] \
                     + 0.5 * x[i-1, j+1] * h[i-2, j+1] \
                     - 4.  * x[i-1, j+1] * h[i-1, j-1] \
                     + 4.  * x[i-1, j+1] * h[i-1, j  ] \
                     + 0.5 * x[i-1, j+1] * h[i,   j-1] \
                     - 0.5 * x[i-1, j+1] * h[i,   j+1] \
                     + 1.  * x[i-1, j+2] * h[i,   j+1] \
                     - 1.  * x[i-1, j+2] * h[i,   j+2] \
                     + 1.  * x[i-1, j  ] * h[i-2, j-1] \
                     - 1.  * x[i-1, j  ] * h[i-2, j+1] \
                     + 4.  * x[i-1, j  ] * h[i-1, j-1] \
                     - 4.  * x[i-1, j  ] * h[i-1, j+1] \
                     - 1.  * x[i-1, j  ] * h[i,   j-2] \
                     + 1.  * x[i-1, j  ] * h[i,   j+2] \
                     - 1.  * x[i+1, j-2] * h[i,   j-2] \
                     + 1.  * x[i+1, j-2] * h[i,   j-1] \
                     - 4.  * x[i+1, j-1] * h[i+1, j+1] \
                     + 4.  * x[i+1, j-1] * h[i+1, j  ] \
                     + 0.5 * x[i+1, j-1] * h[i+2, j-1] \
                     - 0.5 * x[i+1, j-1] * h[i+2, j+1] \
                     - 0.5 * x[i+1, j-1] * h[i,   j-1] \
                     + 0.5 * x[i+1, j-1] * h[i,   j+1] \
                     + 4.  * x[i+1, j+1] * h[i+1, j-1] \
                     - 4.  * x[i+1, j+1] * h[i+1, j  ] \
                     + 0.5 * x[i+1, j+1] * h[i+2, j-1] \
                     - 0.5 * x[i+1, j+1] * h[i+2, j+1] \
                     - 0.5 * x[i+1, j+1] * h[i,   j-1] \
                     + 0.5 * x[i+1, j+1] * h[i,   j+1] \
                     - 1.  * x[i+1, j+2] * h[i,   j+1] \
                     + 1.  * x[i+1, j+2] * h[i,   j+2] \
                     - 4.  * x[i+1, j  ] * h[i+1, j-1] \
                     + 4.  * x[i+1, j  ] * h[i+1, j+1] \
                     - 1.  * x[i+1, j  ] * h[i+2, j-1] \
                     + 1.  * x[i+1, j  ] * h[i+2, j+1] \
                     + 1.  * x[i+1, j  ] * h[i,   j-2] \
                     - 1.  * x[i+1, j  ] * h[i,   j+2] \
                     - 0.5 * x[i+2, j-1] * h[i+1, j-1] \
                     - 0.5 * x[i+2, j-1] * h[i+1, j+1] \
                     + 1.  * x[i+2, j-1] * h[i+1, j  ] \
                     + 0.5 * x[i+2, j+1] * h[i+1, j-1] \
                     + 0.5 * x[i+2, j+1] * h[i+1, j+1] \
                     - 1.  * x[i+2, j+1] * h[i+1, j  ] \
                     - 1.  * x[i,   j-2] * h[i-1, j-2] \
                     + 1.  * x[i,   j-2] * h[i-1, j  ] \
                     + 1.  * x[i,   j-2] * h[i+1, j-2] \
                     - 1.  * x[i,   j-2] * h[i+1, j  ] \
                     + 1.  * x[i,   j-1] * h[i-1, j-2] \
                     - 0.5 * x[i,   j-1] * h[i-1, j-1] \
                     - 0.5 * x[i,   j-1] * h[i-1, j+1] \
                     - 1.  * x[i,   j-1] * h[i+1, j-2] \
                     + 0.5 * x[i,   j-1] * h[i+1, j-1] \
                     + 0.5 * x[i,   j-1] * h[i+1, j+1] \
                     + 0.5 * x[i,   j+1] * h[i-1, j-1] \
                     + 0.5 * x[i,   j+1] * h[i-1, j+1] \
                     - 1.  * x[i,   j+1] * h[i-1, j+2] \
                     - 0.5 * x[i,   j+1] * h[i+1, j-1] \
                     - 0.5 * x[i,   j+1] * h[i+1, j+1] \
                     + 1.  * x[i,   j+1] * h[i+1, j+2] \
                     + 1.  * x[i,   j+2] * h[i-1, j+2] \
                     - 1.  * x[i,   j+2] * h[i-1, j  ] \
                     - 1.  * x[i,   j+2] * h[i+1, j+2] \
                     + 1.  * x[i,   j+2] * h[i+1, j  ] \
                 ) * self.hx_inv * self.hy_inv**3 / 24. \
               + ( \
                     - 1.  * x[i-2, j-1] * h[i-2, j  ] \
                     + 1.  * x[i-2, j-1] * h[i-1, j  ] \
                     + 1.  * x[i-2, j+1] * h[i-2, j  ] \
                     - 1.  * x[i-2, j+1] * h[i-1, j  ] \
                     + 1.  * x[i-2, j  ] * h[i-2, j-1] \
                     - 1.  * x[i-2, j  ] * h[i-2, j+1] \
                     - 1.  * x[i-2, j  ] * h[i,   j-1] \
                     + 1.  * x[i-2, j  ] * h[i,   j+1] \
                     - 0.5 * x[i-1, j-2] * h[i-1, j-1] \
                     - 0.5 * x[i-1, j-2] * h[i+1, j-1] \
                     + 1.  * x[i-1, j-2] * h[i,   j-1] \
                     + 0.5 * x[i-1, j-1] * h[i-1, j-2] \
                     - 0.5 * x[i-1, j-1] * h[i-1, j  ] \
                     - 0.5 * x[i-1, j-1] * h[i+1, j-2] \
                     - 4.  * x[i-1, j-1] * h[i+1, j-1] \
                     + 0.5 * x[i-1, j-1] * h[i+1, j  ] \
                     + 4.  * x[i-1, j-1] * h[i,   j-1] \
                     - 0.5 * x[i-1, j+1] * h[i-1, j+2] \
                     + 0.5 * x[i-1, j+1] * h[i-1, j  ] \
                     + 4.  * x[i-1, j+1] * h[i+1, j+1] \
                     + 0.5 * x[i-1, j+1] * h[i+1, j+2] \
                     - 0.5 * x[i-1, j+1] * h[i+1, j  ] \
                     - 4.  * x[i-1, j+1] * h[i,   j+1] \
                     + 0.5 * x[i-1, j+2] * h[i-1, j+1] \
                     + 0.5 * x[i-1, j+2] * h[i+1, j+1] \
                     - 1.  * x[i-1, j+2] * h[i,   j+1] \
                     - 1.  * x[i-1, j  ] * h[i-2, j-1] \
                     + 1.  * x[i-1, j  ] * h[i-2, j+1] \
                     + 0.5 * x[i-1, j  ] * h[i-1, j-1] \
                     - 0.5 * x[i-1, j  ] * h[i-1, j+1] \
                     + 0.5 * x[i-1, j  ] * h[i+1, j-1] \
                     - 0.5 * x[i-1, j  ] * h[i+1, j+1] \
                     + 0.5 * x[i+1, j-2] * h[i-1, j-1] \
                     + 0.5 * x[i+1, j-2] * h[i+1, j-1] \
                     - 1.  * x[i+1, j-2] * h[i,   j-1] \
                     + 0.5 * x[i+1, j-1] * h[i-1, j-2] \
                     + 4.  * x[i+1, j-1] * h[i-1, j-1] \
                     - 0.5 * x[i+1, j-1] * h[i-1, j  ] \
                     - 0.5 * x[i+1, j-1] * h[i+1, j-2] \
                     + 0.5 * x[i+1, j-1] * h[i+1, j  ] \
                     - 4.  * x[i+1, j-1] * h[i,   j-1] \
                     - 4.  * x[i+1, j+1] * h[i-1, j+1] \
                     - 0.5 * x[i+1, j+1] * h[i-1, j+2] \
                     + 0.5 * x[i+1, j+1] * h[i-1, j  ] \
                     + 0.5 * x[i+1, j+1] * h[i+1, j+2] \
                     - 0.5 * x[i+1, j+1] * h[i+1, j  ] \
                     + 4.  * x[i+1, j+1] * h[i,   j+1] \
                     - 0.5 * x[i+1, j+2] * h[i-1, j+1] \
                     - 0.5 * x[i+1, j+2] * h[i+1, j+1] \
                     + 1.  * x[i+1, j+2] * h[i,   j+1] \
                     - 0.5 * x[i+1, j  ] * h[i-1, j-1] \
                     + 0.5 * x[i+1, j  ] * h[i-1, j+1] \
                     - 0.5 * x[i+1, j  ] * h[i+1, j-1] \
                     + 0.5 * x[i+1, j  ] * h[i+1, j+1] \
                     + 1.  * x[i+1, j  ] * h[i+2, j-1] \
                     - 1.  * x[i+1, j  ] * h[i+2, j+1] \
                     - 1.  * x[i+2, j-1] * h[i+1, j  ] \
                     + 1.  * x[i+2, j-1] * h[i+2, j  ] \
                     + 1.  * x[i+2, j+1] * h[i+1, j  ] \
                     - 1.  * x[i+2, j+1] * h[i+2, j  ] \
                     - 1.  * x[i+2, j  ] * h[i+2, j-1] \
                     + 1.  * x[i+2, j  ] * h[i+2, j+1] \
                     + 1.  * x[i+2, j  ] * h[i,   j-1] \
                     - 1.  * x[i+2, j  ] * h[i,   j+1] \
                     + 1.  * x[i,   j-1] * h[i-2, j  ] \
                     - 1.  * x[i,   j-1] * h[i-1, j-2] \
                     - 4.  * x[i,   j-1] * h[i-1, j-1] \
                     + 1.  * x[i,   j-1] * h[i+1, j-2] \
                     + 4.  * x[i,   j-1] * h[i+1, j-1] \
                     - 1.  * x[i,   j-1] * h[i+2, j  ] \
                     - 1.  * x[i,   j+1] * h[i-2, j  ] \
                     + 4.  * x[i,   j+1] * h[i-1, j+1] \
                     + 1.  * x[i,   j+1] * h[i-1, j+2] \
                     - 4.  * x[i,   j+1] * h[i+1, j+1] \
                     - 1.  * x[i,   j+1] * h[i+1, j+2] \
                     + 1.  * x[i,   j+1] * h[i+2, j  ] \
                 ) * self.hx_inv**3 * self.hy_inv / 24.
 
        return result
    
    
    @cython.boundscheck(False)
    cdef np.float64_t laplace(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j):
        
        cdef np.float64_t result
        
        result = ( \
                   + 1. * x[i-1, j] \
                   - 2. * x[i,   j] \
                   + 1. * x[i+1, j] \
                 ) * self.hx_inv**2 \
               + ( \
                   + 1. * x[i, j-1] \
                   - 2. * x[i, j  ] \
                   + 1. * x[i, j+1] \
                 ) * self.hy_inv**2
 
        return result
    

    @cython.boundscheck(False)
    cdef np.float64_t laplace_ave(self, np.ndarray[np.float64_t, ndim=2] x,
                                        np.uint64_t i, np.uint64_t j):
        
        cdef np.float64_t result
        
        result = 0.25 * ( \
                 ( \
                   + 1. * x[i-1, j-1] \
                   - 2. * x[i,   j-1] \
                   + 1. * x[i+1, j-1] \
                   + 2. * x[i-1, j  ] \
                   - 4. * x[i,   j  ] \
                   + 2. * x[i+1, j  ] \
                   + 1. * x[i-1, j+1] \
                   - 2. * x[i,   j+1] \
                   + 1. * x[i+1, j+1] \
                 ) * self.hx_inv**2 \
               + ( \
                   + 1. * x[i-1, j-1] \
                   - 2. * x[i-1, j  ] \
                   + 1. * x[i-1, j+1] \
                   + 2. * x[i,   j-1] \
                   - 4. * x[i,   j  ] \
                   + 2. * x[i,   j+1] \
                   + 1. * x[i+1, j-1] \
                   - 2. * x[i+1, j  ] \
                   + 1. * x[i+1, j+1] \
                 ) * self.hy_inv**2 \
               )
 
        return result
    

    @cython.boundscheck(False)
    cpdef laplace_vec(self, Vec X, Vec Y, np.float64_t sign):
    
        cdef np.uint64_t ix, iy, jx, jy, i, j
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(X, self.localX)
        
        x = self.da1.getVecArray(self.localX)
        y = self.da1.getVecArray(Y)
        
        cdef np.ndarray[np.float64_t, ndim=2] tx = x[...]
        cdef np.ndarray[np.float64_t, ndim=2] ty = y[...]
        
        for j in np.arange(ys, ye):
            jx = j-ys+2
            jy = j-ys
            
            for i in np.arange(xs, xe):
                ix = i-xs+2
                iy = i-xs
                
                ty[iy, jy] = sign * self.laplace(tx, ix, jx)
#                 ty[iy, jy] = sign * self.laplace_ave(tx, ix, jx)
    
    
    
    @cython.boundscheck(False)
    cpdef np.float64_t dx(self, Vec X, Vec Y, np.float64_t sign):
    
        cdef np.uint64_t ix, iy, jx, jy, i, j
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(X, self.localX)
        
        x = self.da1.getVecArray(self.localX)
        y = self.da1.getVecArray(Y)
        
        cdef np.ndarray[np.float64_t, ndim=2] tx = x[...]
        cdef np.ndarray[np.float64_t, ndim=2] ty = y[...]
        
        for j in np.arange(ys, ye):
            jx = j-ys+2
            jy = j-ys
            
            for i in np.arange(xs, xe):
                ix = i-xs+2
                iy = i-xs
                
#                 ty[iy, jy] = sign * (tx[ix+1, jx] - tx[ix, jx]) * self.hx_inv
    
                ty[iy, jy] = sign * (tx[ix+1, jx] - tx[ix-1, jx]) * 0.5 * self.hx_inv
    
#                 ty[iy, jy] = sign * 0.25 * ( \
#                                         + 1. * (tx[ix+1, jx-1] - tx[ix-1, jx-1]) \
#                                         + 2. * (tx[ix+1, jx  ] - tx[ix-1, jx  ]) \
#                                         + 1. * (tx[ix+1, jx+1] - tx[ix-1, jx+1]) \
#                                       ) * 0.5 * self.hx_inv
        
    
    @cython.boundscheck(False)
    cpdef np.float64_t dy(self, Vec X, Vec Y, np.float64_t sign):
    
        cdef np.uint64_t ix, iy, jx, jy, i, j
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(X, self.localX)
        
        x = self.da1.getVecArray(self.localX)
        y = self.da1.getVecArray(Y)
        
        cdef np.ndarray[np.float64_t, ndim=2] tx = x[...]
        cdef np.ndarray[np.float64_t, ndim=2] ty = y[...]
        
        for j in np.arange(ys, ye):
            jx = j-ys+2
            jy = j-ys
            
            for i in np.arange(xs, xe):
                ix = i-xs+2
                iy = i-xs
                
#                 ty[iy, jy] = sign * (tx[ix, jx+1] - tx[ix, jx]) * self.hy_inv
    
                ty[iy, jy] = sign * (tx[ix, jx+1] - tx[ix, jx-1]) * 0.5 * self.hy_inv
    
#                 ty[iy, jy] = sign * 0.25 * ( \
#                                         + 1. * (tx[ix-1, jx+1] - tx[ix-1, jx-1]) \
#                                         + 2. * (tx[ix,   jx+1] - tx[ix,   jx-1]) \
#                                         + 1. * (tx[ix+1, jx+1] - tx[ix+1, jx-1]) \
#                                       ) * 0.5 * self.hy_inv
    
    
    @cython.boundscheck(False)
    cdef np.float64_t dt(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        return self.average(x, i, j) * self.ht_inv
    
    
    @cython.boundscheck(False)
    cdef np.float64_t average(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j):
        '''
        Average
        '''
        
        cdef np.float64_t result
        
        result = ( \
                   + 1. * x[i-1, j-1] \
                   + 2. * x[i-1, j  ] \
                   + 1. * x[i-1, j+1] \
                   + 2. * x[i,   j-1] \
                   + 4. * x[i,   j  ] \
                   + 2. * x[i,   j+1] \
                   + 1. * x[i+1, j-1] \
                   + 2. * x[i+1, j  ] \
                   + 1. * x[i+1, j+1] \
                 ) / 16.
        
        return result


    @cython.boundscheck(False)
    cdef np.float64_t dt_laplace(self, np.ndarray[np.float64_t, ndim=2] x,
                                       np.uint64_t i, np.uint64_t j):
        
        cdef np.float64_t result
        
        result = 0.25 * ( \
                 ( \
                   + 1. * x[i-1, j-1] \
                   - 2. * x[i,   j-1] \
                   + 1. * x[i+1, j-1] \
                   + 2. * x[i-1, j  ] \
                   - 4. * x[i,   j  ] \
                   + 2. * x[i+1, j  ] \
                   + 1. * x[i-1, j+1] \
                   - 2. * x[i,   j+1] \
                   + 1. * x[i+1, j+1] \
                 ) * self.hx_inv**2 * self.ht_inv \
               + ( \
                   + 1. * x[i-1, j-1] \
                   - 2. * x[i-1, j  ] \
                   + 1. * x[i-1, j+1] \
                   + 2. * x[i,   j-1] \
                   - 4. * x[i,   j  ] \
                   + 2. * x[i,   j+1] \
                   + 1. * x[i+1, j-1] \
                   - 2. * x[i+1, j  ] \
                   + 1. * x[i+1, j+1] \
                 ) * self.hy_inv**2 * self.ht_inv \
               )
 
        return result
    

