# cython: profile=True
'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport Vec


cdef class PETScDerivatives(object):
    '''
    Cython Implementation of MHD Discretisation
    '''
    
    
    def __cinit__(self, object da1, np.uint64_t nx, np.uint64_t ny,
                       double ht, double hx, double hy):
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
        
        
        # create local vectors
        self.localX = da1.createLocalVec()
        self.localY = da1.createLocalVec()
        
        
    
    @cython.boundscheck(False)
    cdef double arakawa(self, double[:,:] x, double[:,:] h, int i, int j):
        '''
        MHD Derivative: Arakawa Bracket
        '''
        
        cdef double jpp, jpc, jcp, result
        
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
        
        result = (jpp + jpc + jcp) * self.hx_inv * self.hy_inv / 12.
        
        return result
    
    
    @cython.boundscheck(False)
    cpdef arakawa_vec(self, Vec X, Vec Y, Vec A):
        '''
        MHD Derivative: Arakawa Bracket
        '''
        
        cdef int i, j, stencil
        cdef int ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        cdef double jpp, jpc, jcp, fac

        (xs, xe), (ys, ye) = self.da1.getRanges()
        stencil = self.da1.getStencilWidth()
        
        fac = self.hx_inv * self.hy_inv / 12.
        
        self.da1.globalToLocal(X, self.localX)
        self.da1.globalToLocal(Y, self.localY)
        
        cdef double[:,:] a = self.da1.getVecArray(A)[...]
        cdef double[:,:] x = self.da1.getVecArray(self.localX)[...]
        cdef double[:,:] y = self.da1.getVecArray(self.localY)[...]
        
        
        for i in range(xs, xe):
            ix = i-xs+stencil
            iy = i-xs
             
            for j in range(ys, ye):
                jx = j-ys+stencil
                jy = j-ys
                 
                jpp = (x[ix+1, jx  ] - x[ix-1, jx  ]) * (y[ix,   jx+1] - y[ix,   jx-1]) \
                    - (x[ix,   jx+1] - x[ix,   jx-1]) * (y[ix+1, jx  ] - y[ix-1, jx  ])
                
                jpc = x[ix+1, jx  ] * (y[ix+1, jx+1] - y[ix+1, jx-1]) \
                    - x[ix-1, jx  ] * (y[ix-1, jx+1] - y[ix-1, jx-1]) \
                    - x[ix,   jx+1] * (y[ix+1, jx+1] - y[ix-1, jx+1]) \
                    + x[ix,   jx-1] * (y[ix+1, jx-1] - y[ix-1, jx-1])
                
                jcp = x[ix+1, jx+1] * (y[ix,   jx+1] - y[ix+1, jx  ]) \
                    - x[ix-1, jx-1] * (y[ix-1, jx  ] - y[ix,   jx-1]) \
                    - x[ix-1, jx+1] * (y[ix,   jx+1] - y[ix-1, jx  ]) \
                    + x[ix+1, jx-1] * (y[ix+1, jx  ] - y[ix,   jx-1])
                
                a[iy, jy] = fac * (jpp + jpc + jcp)
        
        
    
    @cython.boundscheck(False)
    cdef double laplace(self, double[:,:] x, int i, int j):
        
        cdef double result
        
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
    cpdef laplace_vec(self, Vec X, Vec Y, double sign):
    
        cdef int i, j, stencil
        cdef int ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        stencil = self.da1.getStencilWidth()
        
        self.da1.globalToLocal(X, self.localX)
        
        x = self.da1.getVecArray(self.localX)
        y = self.da1.getVecArray(Y)
        
        cdef double[:,:] tx = x[...]
        cdef double[:,:] ty = y[...]
        
        for i in range(xs, xe):
            ix = i-xs+stencil
            iy = i-xs
             
            for j in range(ys, ye):
                jx = j-ys+stencil
                jy = j-ys
                
                ty[iy, jy] = sign * self.laplace(tx, ix, jx)
    
    
    
    @cython.boundscheck(False)
    cpdef double dx(self, Vec X, Vec Y, double sign):
    
        cdef int ix, iy, jx, jy, i, j
        cdef int xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(X, self.localX)
        
        x = self.da1.getVecArray(self.localX)
        y = self.da1.getVecArray(Y)
        
        cdef double[:,:] tx = x[...]
        cdef double[:,:] ty = y[...]
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                ty[iy, jy] = sign * (tx[ix+1, jx] - tx[ix-1, jx]) * 0.5 * self.hx_inv
    
    
    @cython.boundscheck(False)
    cpdef double dy(self, Vec X, Vec Y, double sign):
    
        cdef int ix, iy, jx, jy, i, j
        cdef int xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(X, self.localX)
        
        x = self.da1.getVecArray(self.localX)
        y = self.da1.getVecArray(Y)
        
        cdef double[:,:] tx = x[...]
        cdef double[:,:] ty = y[...]
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                ty[iy, jy] = sign * (tx[ix, jx+1] - tx[ix, jx-1]) * 0.5 * self.hy_inv
    
    
    @cython.boundscheck(False)
    cdef double dt(self, double[:,:] x, int i, int j):
        '''
        Time Derivative
        '''
        
        return x[i, j] * self.ht_inv
    
    