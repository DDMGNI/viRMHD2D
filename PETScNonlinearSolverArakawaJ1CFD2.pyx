# cython: profile=True
'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport Mat, Vec

from PETScDerivatives import PETScDerivatives


cdef class PETScSolver(object):
    '''
    The PETScSolver class implements a nonlinear solver for the reduced MHD system
    built on top of the PETSc SNES module.
    '''
    
    def __init__(self, object da1, object da4,
                 int nx, int ny,
                 double ht, double hx, double hy,
                 object pc=None):
        '''
        Constructor
        '''
        
        # distributed arrays
        self.da1 = da1
        self.da4 = da4
        
        # preconditioner
        self.pc = pc
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        self.ht_inv = 1. / ht
        self.hx_inv = 1. / hx
        self.hy_inv = 1. / hy
        
        
        # create history vector
        self.Xp = self.da4.createGlobalVec()
        self.Xh = self.da4.createGlobalVec()
        self.Yd = self.da4.createGlobalVec()
        
        self.Ap = self.da1.createGlobalVec()
        self.Jp = self.da1.createGlobalVec()
        self.Pp = self.da1.createGlobalVec()
        self.Op = self.da1.createGlobalVec()
        
        self.Ah = self.da1.createGlobalVec()
        self.Jh = self.da1.createGlobalVec()
        self.Ph = self.da1.createGlobalVec()
        self.Oh = self.da1.createGlobalVec()
        
        self.Aa = self.da1.createGlobalVec()
        self.Ja = self.da1.createGlobalVec()
        self.Pa = self.da1.createGlobalVec()
        self.Oa = self.da1.createGlobalVec()
        
        # create working vector
        self.YA = self.da1.createGlobalVec()
        self.YJ = self.da1.createGlobalVec()
        self.YP = self.da1.createGlobalVec()
        self.YO = self.da1.createGlobalVec()
        
        self.Ad = self.da1.createGlobalVec()
        self.Jd = self.da1.createGlobalVec()
        self.Pd = self.da1.createGlobalVec()
        self.Od = self.da1.createGlobalVec()
        
        # create temporary vectors
        self.T  = self.da1.createGlobalVec()
        self.T1 = self.da1.createGlobalVec()
        self.T2 = self.da1.createGlobalVec()
        self.T3 = self.da1.createGlobalVec()
        self.T4 = self.da1.createGlobalVec()
        
        # create local vectors
        self.localAa = self.da1.createLocalVec()
        self.localJa = self.da1.createLocalVec()
        self.localPa = self.da1.createLocalVec()
        self.localOa = self.da1.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETScDerivatives(da1, nx, ny, ht, hx, hy)
        
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
        x = self.da4.getVecArray(self.Xh)
         
        self.da1.getVecArray(self.Ah)[:,:] = x[:,:,0]
        self.da1.getVecArray(self.Jh)[:,:] = x[:,:,1]
        self.da1.getVecArray(self.Ph)[:,:] = x[:,:,2]
        self.da1.getVecArray(self.Oh)[:,:] = x[:,:,3]
        
    
    def update_previous(self, Vec X):
        X.copy(self.Xp)
        
        x = self.da4.getVecArray(self.Xp)
         
        self.da1.getVecArray(self.Ap)[:,:] = x[:,:,0]
        self.da1.getVecArray(self.Jp)[:,:] = x[:,:,1]
        self.da1.getVecArray(self.Pp)[:,:] = x[:,:,2]
        self.da1.getVecArray(self.Op)[:,:] = x[:,:,3]
        
        self.da1.getVecArray(self.Aa)[:,:] = 0.5 * (self.da1.getVecArray(self.Ap)[:,:] + self.da1.getVecArray(self.Ah)[:,:])
        self.da1.getVecArray(self.Ja)[:,:] = 0.5 * (self.da1.getVecArray(self.Jp)[:,:] + self.da1.getVecArray(self.Jh)[:,:])
        self.da1.getVecArray(self.Pa)[:,:] = 0.5 * (self.da1.getVecArray(self.Pp)[:,:] + self.da1.getVecArray(self.Ph)[:,:])
        self.da1.getVecArray(self.Oa)[:,:] = 0.5 * (self.da1.getVecArray(self.Op)[:,:] + self.da1.getVecArray(self.Oh)[:,:])
        
        self.da1.globalToLocal(self.Aa, self.localAa)
        self.da1.globalToLocal(self.Ja, self.localJa)
        self.da1.globalToLocal(self.Pa, self.localPa)
        self.da1.globalToLocal(self.Oa, self.localOa)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef int i, j, stencil
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        stencil = self.da1.getStencilWidth()
        
        cdef double[:,:] A_ave = self.da1.getVecArray(self.localAa)[...]
        cdef double[:,:] J_ave = self.da1.getVecArray(self.localJa)[...]
        cdef double[:,:] P_ave = self.da1.getVecArray(self.localPa)[...]
        cdef double[:,:] O_ave = self.da1.getVecArray(self.localOa)[...]
        
        
        cdef double arak_fac = 0.5 * self.ht * self.hx_inv * self.hy_inv / 12.
        cdef double lapx_fac = self.hx_inv**2
        cdef double lapy_fac = self.hy_inv**2
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in range(xs, xe):
            ix = i-xs+stencil
            
            for j in range(ys, ye):
                jx = j-ys+stencil
                
                row.index = (i,j)
                
                # magnetic potential
                # dA/dt + [P, A] = - R_J
                row.field = 0
                
                # dA/dt + [P, dA] 
                col.field = 0
                for index, value in [
                        ((i-1, j-1), + (P_ave[ix-1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), + (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                     + (P_ave[ix-1, jx+1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), + (P_ave[ix,   jx+1] - P_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), - (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                     - (P_ave[ix+1, jx-1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j  ), 1.),
                        ((i,   j+1), + (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                     + (P_ave[ix+1, jx+1] - P_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), - (P_ave[ix+1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), - (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                     - (P_ave[ix+1, jx+1] - P_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), - (P_ave[ix,   jx+1] - P_ave[ix+1, jx  ]) * arak_fac),
                    ]:
  
                    col.index = index
                    A.setValueStencil(row, col, value)
                  
                  
                # + [dP, A]
                col.field = 2
                for index, value in [
                        ((i-1, j-1), - (A_ave[ix-1, jx  ] - A_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), - (A_ave[ix,   jx+1] - A_ave[ix,   jx-1]) * arak_fac \
                                     - (A_ave[ix-1, jx+1] - A_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), - (A_ave[ix,   jx+1] - A_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), + (A_ave[ix+1, jx  ] - A_ave[ix-1, jx  ]) * arak_fac \
                                     + (A_ave[ix+1, jx-1] - A_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j+1), - (A_ave[ix+1, jx  ] - A_ave[ix-1, jx  ]) * arak_fac \
                                     - (A_ave[ix+1, jx+1] - A_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), + (A_ave[ix+1, jx  ] - A_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), + (A_ave[ix,   jx+1] - A_ave[ix,   jx-1]) * arak_fac \
                                     + (A_ave[ix+1, jx+1] - A_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), + (A_ave[ix,   jx+1] - A_ave[ix+1, jx  ]) * arak_fac),
                    ]:
  
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # current density
                # - Delta A - J = - R_A
                row.field = 1
                
                # - Delta A 
                col.field = 0
                for index, value in [
                    ((i,   j-1),                 - 1. * lapy_fac),
                    ((i-1, j  ), - 1. * lapx_fac                ),
                    ((i,   j  ), + 2. * lapx_fac + 2. * lapy_fac),
                    ((i+1, j  ), - 1. * lapx_fac                ),
                    ((i,   j+1),                 - 1. * lapy_fac),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, - value)
            
                # - J
                col.field = 1
                col.index = (i,j)
                A.setValueStencil(row, col, 1.)
                
                
                
                # streaming function
                # - Delta P - O = - R_P
                row.field = 2
                
                # - Delta P
                col.field = 2
                for index, value in [
                    ((i,   j-1),                 - 1. * lapy_fac),
                    ((i-1, j  ), - 1. * lapx_fac                ),
                    ((i,   j  ), + 2. * lapx_fac + 2. * lapy_fac),
                    ((i+1, j  ), - 1. * lapx_fac                ),
                    ((i,   j+1),                 - 1. * lapy_fac),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, - value)
            
                # - O
                col.field = 3
                col.index = (i,j)
                A.setValueStencil(row, col, 1.)
                
                
                
                # vorticity
                # dO/dt + [P, O] + [J, A] = - R_O
                row.field = 3
                
                # + [J, dA]
                col.field = 0
                for index, value in [
                        ((i-1, j-1), + (J_ave[ix-1, jx  ] - J_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), + (J_ave[ix,   jx+1] - J_ave[ix,   jx-1]) * arak_fac \
                                     + (J_ave[ix-1, jx+1] - J_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), + (J_ave[ix,   jx+1] - J_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), - (J_ave[ix+1, jx  ] - J_ave[ix-1, jx  ]) * arak_fac \
                                     - (J_ave[ix+1, jx-1] - J_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j+1), + (J_ave[ix+1, jx  ] - J_ave[ix-1, jx  ]) * arak_fac \
                                     + (J_ave[ix+1, jx+1] - J_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), - (J_ave[ix+1, jx  ] - J_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), - (J_ave[ix,   jx+1] - J_ave[ix,   jx-1]) * arak_fac \
                                     - (J_ave[ix+1, jx+1] - J_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), - (J_ave[ix,   jx+1] - J_ave[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                
                # + [dJ, A]
                col.field = 1
                for index, value in [
                        ((i-1, j-1), - (A_ave[ix-1, jx  ] - A_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), - (A_ave[ix,   jx+1] - A_ave[ix,   jx-1]) * arak_fac \
                                     - (A_ave[ix-1, jx+1] - A_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), - (A_ave[ix,   jx+1] - A_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), + (A_ave[ix+1, jx  ] - A_ave[ix-1, jx  ]) * arak_fac \
                                     + (A_ave[ix+1, jx-1] - A_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j+1), - (A_ave[ix+1, jx  ] - A_ave[ix-1, jx  ]) * arak_fac \
                                     - (A_ave[ix+1, jx+1] - A_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), + (A_ave[ix+1, jx  ] - A_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), + (A_ave[ix,   jx+1] - A_ave[ix,   jx-1]) * arak_fac \
                                     + (A_ave[ix+1, jx+1] - A_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), + (A_ave[ix,   jx+1] - A_ave[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # + [dP, O]
                col.field = 2
                for index, value in [
                        ((i-1, j-1), - (O_ave[ix-1, jx  ] - O_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), - (O_ave[ix,   jx+1] - O_ave[ix,   jx-1]) * arak_fac \
                                     - (O_ave[ix-1, jx+1] - O_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), - (O_ave[ix,   jx+1] - O_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), + (O_ave[ix+1, jx  ] - O_ave[ix-1, jx  ]) * arak_fac \
                                     + (O_ave[ix+1, jx-1] - O_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j+1), - (O_ave[ix+1, jx  ] - O_ave[ix-1, jx  ]) * arak_fac \
                                     - (O_ave[ix+1, jx+1] - O_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), + (O_ave[ix+1, jx  ] - O_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), + (O_ave[ix,   jx+1] - O_ave[ix,   jx-1]) * arak_fac \
                                     + (O_ave[ix+1, jx+1] - O_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), + (O_ave[ix,   jx+1] - O_ave[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # dO/dt + [P, dO] 
                col.field = 3
                for index, value in [
                        ((i-1, j-1), + (P_ave[ix-1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), + (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                     + (P_ave[ix-1, jx+1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), + (P_ave[ix,   jx+1] - P_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), - (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                     - (P_ave[ix+1, jx-1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j  ), 1.),
                        ((i,   j+1), + (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                     + (P_ave[ix+1, jx+1] - P_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), - (P_ave[ix+1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), - (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                     - (P_ave[ix+1, jx+1] - P_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), - (P_ave[ix,   jx+1] - P_ave[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
        A.assemble()
        
    
    @cython.boundscheck(False)
    def mult(self, Mat mat, Vec X, Vec Y):
        
        if self.pc == None:
            xd = self.da4.getVecArray(X)
        else:
            self.pc.solve(X, self.Yd)
            xd = self.da4.getVecArray(self.Yd)
        
        self.da1.getVecArray(self.Ad)[:,:] = xd[:,:,0]
        self.da1.getVecArray(self.Jd)[:,:] = xd[:,:,1]
        self.da1.getVecArray(self.Pd)[:,:] = xd[:,:,2]
        self.da1.getVecArray(self.Od)[:,:] = xd[:,:,3]

        
        y = self.da4.getVecArray(Y)
         
        # magnetic potential
        self.derivatives.arakawa_vec(self.Pa, self.Ad, self.T1)
        self.derivatives.arakawa_vec(self.Pd, self.Aa, self.T2)
        
        y[:,:,0] = self.da1.getVecArray(self.Ad)[:,:] \
                 + 0.5*self.ht * self.da1.getVecArray(self.T1)[:,:] \
                 + 0.5*self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        # current density
        self.derivatives.laplace_vec(self.Ad, self.YJ, +1.)

        y[:,:,1] = self.da1.getVecArray(self.Jd)[:,:] \
                 + self.da1.getVecArray(self.YJ)[:,:]
         
        # streaming function
        self.derivatives.laplace_vec(self.Pd, self.YP, +1.)

        y[:,:,2] = self.da1.getVecArray(self.Od)[:,:] \
                 + self.da1.getVecArray(self.YP)[:,:]
 
        # vorticity
        self.derivatives.arakawa_vec(self.Pa, self.Od, self.T1)
        self.derivatives.arakawa_vec(self.Pd, self.Oa, self.T2)
        self.derivatives.arakawa_vec(self.Ja, self.Ad, self.T3)
        self.derivatives.arakawa_vec(self.Jd, self.Aa, self.T4)
        
        y[:,:,3] = self.da1.getVecArray(self.Od)[:,:] \
                 + 0.5*self.ht * self.da1.getVecArray(self.T1)[:,:] \
                 + 0.5*self.ht * self.da1.getVecArray(self.T2)[:,:] \
                 + 0.5*self.ht * self.da1.getVecArray(self.T3)[:,:] \
                 + 0.5*self.ht * self.da1.getVecArray(self.T4)[:,:]

   
    def snes_function(self, SNES snes, Vec X, Vec Y):
        self.update_previous(X)
        self.function(Y)
        
    
    @cython.boundscheck(False)
    def function(self, Vec Y):

        y = self.da4.getVecArray(Y)
         
        # magnetic potential
        self.derivatives.arakawa_vec(self.Pa, self.Aa, self.T1)
        
        y[:,:,0] = self.da1.getVecArray(self.Ap)[:,:] \
                 - self.da1.getVecArray(self.Ah)[:,:] \
                 + self.ht * self.da1.getVecArray(self.T1)[:,:]
        
        # current density
        self.derivatives.laplace_vec(self.Ap, self.YJ, +1.)

        y[:,:,1] = self.da1.getVecArray(self.Jp)[:,:] \
                 + self.da1.getVecArray(self.YJ)[:,:]
         
        # streaming function
        self.derivatives.laplace_vec(self.Pp, self.YP, +1.)

        y[:,:,2] = self.da1.getVecArray(self.Op)[:,:] \
                 + self.da1.getVecArray(self.YP)[:,:]
 
        # vorticity
        self.derivatives.arakawa_vec(self.Pa, self.Oa, self.T1)
        self.derivatives.arakawa_vec(self.Ja, self.Aa, self.T2)
        
        y[:,:,3] = self.da1.getVecArray(self.Op)[:,:] \
                 - self.da1.getVecArray(self.Oh)[:,:] \
                 + self.ht * self.da1.getVecArray(self.T1)[:,:] \
                 + self.ht * self.da1.getVecArray(self.T2)[:,:]
        
