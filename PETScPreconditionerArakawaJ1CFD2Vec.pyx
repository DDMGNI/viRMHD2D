'''
Created on Apr 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

from datetime import datetime

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport Mat, Vec

from PETScDerivatives import PETScDerivatives
from PETScPoissonCFD2 import PETScPoisson


cdef class PETScPreconditioner(object):
    '''
    The PETScPreconditioner class implements a preconditioner for the reduced MHD system
    built on top of the PETSc SNES module.
    '''
    
    def __init__(self, object da1, object da4,
                 int nx, int ny,
                 double ht, double hx, double hy,
                 double skin_depth=0.):
        '''
        Constructor
        '''
        
        # distributed arrays
        self.da1 = da1
        self.da4 = da4
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        self.ht_inv = 1. / ht
        self.hx_inv = 1. / hx
        self.hy_inv = 1. / hy
        
        # electron skin depth
        self.de = skin_depth
        
        self.lapx_fac = self.de**2 * self.hx_inv**2
        self.lapy_fac = self.de**2 * self.hy_inv**2
        
        # jacobi solver
        self.jacobi_max_it = 3
        
        # create solver vectors
        self.F  = self.da4.createGlobalVec()
        self.L  = self.da1.createGlobalVec()
        self.T  = self.da1.createGlobalVec()
        
        self.FA = self.da1.createGlobalVec()
        self.FJ = self.da1.createGlobalVec()
        self.FP = self.da1.createGlobalVec()
        self.FO = self.da1.createGlobalVec()
        
        self.T1 = self.da1.createGlobalVec()
        self.T2 = self.da1.createGlobalVec()
        self.T3 = self.da1.createGlobalVec()
        self.T4 = self.da1.createGlobalVec()
        
        # create data and history vectors
        self.Xd = self.da4.createGlobalVec()
        
        self.Ad = self.da1.createGlobalVec()
        self.Jd = self.da1.createGlobalVec()
        self.Pd = self.da1.createGlobalVec()
        self.Od = self.da1.createGlobalVec()
        
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
        
        # create local vectors
        self.localQ  = self.da1.createLocalVec()
        self.localT  = self.da1.createLocalVec()
        
        self.localAa = self.da1.createLocalVec()
        self.localPa = self.da1.createLocalVec()
        
        self.localAd = self.da1.createLocalVec()
        self.localPd = self.da1.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETScDerivatives(da1, nx, ny, ht, hx, hy)
        
        # create Nullspace
        self.poisson_nullspace = PETSc.NullSpace().create(constant=True)
        
        # initialise rhs and matrix for Poisson solver
        self.Pb = self.da1.createGlobalVec()
        self.Pm = self.da1.createMat()
        self.Pm.setOption(self.Pm.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.Pm.setUp()
        self.Pm.setNullSpace(self.poisson_nullspace)
        
        # create Poisson solver object and build matrix
        self.petsc_poisson = PETScPoisson(self.da1, self.nx, self.ny, self.hx, self.hy)
        self.petsc_poisson.formMat(self.Pm)
        
        # create linear Poisson solver
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.Pm)
        self.poisson_ksp.setType('cg')
        self.poisson_ksp.getPC().setType('hypre')
        self.poisson_ksp.setUp()

        # initialise rhs and matrixfree matrix for preconditioner
        self.Qb = self.da1.createGlobalVec()
        self.Qm = PETSc.Mat().createPython([self.Qb.getSizes(), self.Qb.getSizes()], 
                                            context=self,
                                            comm=PETSc.COMM_WORLD)
        self.Qm.setUp()

        # create linear parabolic solver
        self.parabol_ksp = PETSc.KSP().create()
        self.parabol_ksp.setFromOptions()
        if self.de != 0.:
            self.QA = self.da1.createMat()
            self.QA.setOption(self.QA.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            self.QA.setUp()
            self.formMat(self.QA)
        
            self.parabol_ksp.setOperators(self.Qm, self.QA)
            self.poisson_ksp.getPC().setType('hypre')
        else:
            self.parabol_ksp.setOperators(self.Qm)
            self.parabol_ksp.getPC().setType('none')

#         self.parabol_ksp.setNormType(self.parabol_ksp.NormType.NORM_NONE)
#         self.parabol_ksp.setType('gmres')
        self.parabol_ksp.setType('cg')
        self.parabol_ksp.setUp()
        
    
    def set_tolerances(self, poisson_rtol=0., poisson_atol=0., poisson_max_it=0,
                             parabol_rtol=0., parabol_atol=0., parabol_max_it=0,
                             jacobi_max_it=0):
        
        if poisson_rtol > 0.:
            self.poisson_ksp.setTolerances(rtol=poisson_rtol)
            
        if poisson_atol > 0.:
            self.poisson_ksp.setTolerances(atol=poisson_atol)
            
        if poisson_max_it > 0:
            self.poisson_ksp.setTolerances(max_it=poisson_max_it)
            
        if parabol_rtol > 0.:
            self.parabol_ksp.setTolerances(rtol=parabol_rtol)
            
        if parabol_atol > 0.:
            self.parabol_ksp.setTolerances(atol=parabol_atol)
            
        if parabol_max_it > 0:
            self.parabol_ksp.setTolerances(max_it=parabol_max_it)
            
        if jacobi_max_it > 0:
            self.jacobi_max_it = jacobi_max_it
    
    
    def update_history(self, Vec Ah, Vec Jh, Vec Ph, Vec Oh):
        Ah.copy(self.Ah)
        Jh.copy(self.Jh)
        Ph.copy(self.Ph)
        Oh.copy(self.Oh)
        
    
    def update_previous(self, Vec Ap, Vec Jp, Vec Pp, Vec Op):
        Ap.copy(self.Ap)
        Jp.copy(self.Jp)
        Pp.copy(self.Pp)
        Op.copy(self.Op)
        
        self.da1.getVecArray(self.Aa)[:,:] = 0.5 * (self.da1.getVecArray(self.Ap)[:,:] + self.da1.getVecArray(self.Ah)[:,:])
        self.da1.getVecArray(self.Ja)[:,:] = 0.5 * (self.da1.getVecArray(self.Jp)[:,:] + self.da1.getVecArray(self.Jh)[:,:])
        self.da1.getVecArray(self.Pa)[:,:] = 0.5 * (self.da1.getVecArray(self.Pp)[:,:] + self.da1.getVecArray(self.Ph)[:,:])
        self.da1.getVecArray(self.Oa)[:,:] = 0.5 * (self.da1.getVecArray(self.Op)[:,:] + self.da1.getVecArray(self.Oh)[:,:])
        
    
    def update_function(self, Vec F):
        F.copy(self.F)
        
        f = self.da4.getVecArray(self.F)
        
        self.da1.getVecArray(self.FA)[:,:] = f[:,:,0]
        self.da1.getVecArray(self.FJ)[:,:] = f[:,:,1]
        self.da1.getVecArray(self.FP)[:,:] = f[:,:,2]
        self.da1.getVecArray(self.FO)[:,:] = f[:,:,3]
        
    
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def solve(self, Vec X, Vec Y):
        
        self.update_function(X)
        
        
        self.derivatives.arakawa_vec(self.Pa, self.FP, self.T1)
        self.derivatives.arakawa_vec(self.Aa, self.FJ, self.T2)
        
        self.da1.getVecArray(self.Pb)[:,:] = self.da1.getVecArray(self.FO)[:,:] \
                                           - self.da1.getVecArray(self.FP)[:,:] \
                                           + 0.5*self.ht * self.da1.getVecArray(self.T1)[:,:] \
                                           - 0.5*self.ht * self.da1.getVecArray(self.T2)[:,:]
        
#         self.L.set(0.)
        self.poisson_nullspace.remove(self.Pb)
        self.poisson_ksp.solve(self.Pb, self.L)
        
        
        self.Ad.set(0.)
        self.Jd.set(0.)
#         self.Pd.set(0.)
        self.Od.set(0.)
        self.L.copy(self.Pd)
        
        for k in range(self.jacobi_max_it):
            self.derivatives.arakawa_vec(self.Pa, self.Pd, self.T)
#             self.T.scale(0.5*self.ht)
            
            self.derivatives.arakawa_vec(self.Aa, self.L,  self.T1)
            self.derivatives.arakawa_vec(self.Pa, self.Ad, self.T2)
            self.derivatives.arakawa_vec(self.Aa, self.T,  self.T3)
            
            
            self.da1.getVecArray(self.Qb)[:,:] = self.da1.getVecArray(self.FA)[:,:] \
                                               + 0.5*self.ht * self.da1.getVecArray(self.T1)[:,:] \
                                               - 0.5*self.ht * self.da1.getVecArray(self.T2)[:,:] \
                                               - 0.5*self.ht * 0.5*self.ht * self.da1.getVecArray(self.T3)[:,:]

            if self.de != 0.:
                self.derivatives.arakawa_vec(self.Pd, self.Ja, self.T1)
                self.derivatives.arakawa_vec(self.Pa, self.Jd, self.T2)
                   
                self.da1.getVecArray(self.Qb)[:,:] -= self.de**2 * 0.5*self.ht * self.da1.getVecArray(self.T1)[:,:] \
                                                    + self.de**2 * 0.5*self.ht * self.da1.getVecArray(self.T2)[:,:] \
                                                    + self.de**2 * self.da1.getVecArray(self.Jd)[:,:]

#                 self.derivatives.laplace_vec(self.Ad, self.T3, -1.)
#                 self.derivatives.arakawa_vec(self.Pd, self.Ja, self.T1)
#                 self.derivatives.arakawa_vec(self.Pa, self.T3, self.T2)
#                 self.derivatives.arakawa_vec(self.Pa, self.FJ, self.T3)
#  
#                 self.da1.getVecArray(self.Qb)[:,:] += self.de**2 * 0.5*self.ht * self.da1.getVecArray(self.T1)[:,:] \
#                                                     + self.de**2 * 0.5*self.ht * self.da1.getVecArray(self.T2)[:,:] \
#                                                     + self.de**2 * 0.5*self.ht * self.da1.getVecArray(self.T3)[:,:] \
#                                                     - self.de**2 * self.da1.getVecArray(self.FJ)[:,:]
                
                self.derivatives.arakawa_vec(self.Aa, self.Ad, self.T1)
                self.derivatives.arakawa_vec(self.Aa, self.T1, self.T2)
                
                self.da1.getVecArray(self.Qb)[:,:] += 0.5*self.ht*0.5*self.ht * self.da1.getVecArray(self.T2)[:,:]
                
            
            self.parabol_ksp.solve(self.Qb, self.Ad)
            
            self.derivatives.arakawa_vec(self.Aa, self.Ad, self.T4)
            
            self.da1.getVecArray(self.Pd)[:,:] = self.da1.getVecArray(self.L)[:,:] \
                                               - 0.5*self.ht * self.da1.getVecArray(self.T)[:,:] \
                                               + 0.5*self.ht * self.da1.getVecArray(self.T4)[:,:]
            
            self.derivatives.laplace_vec(self.Pd, self.Od, -1.)
            self.derivatives.laplace_vec(self.Ad, self.Jd, -1.)
            
            self.Od.axpy(1., self.FP)
            self.Jd.axpy(1., self.FJ)
        
        
        y = self.da4.getVecArray(Y)
        
        y[:,:,0] = self.da1.getVecArray(self.Ad)[:,:]
        y[:,:,1] = self.da1.getVecArray(self.Jd)[:,:]
        y[:,:,2] = self.da1.getVecArray(self.Pd)[:,:]
        y[:,:,3] = self.da1.getVecArray(self.Od)[:,:]
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef int i, j, stencil
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        stencil = self.da1.getStencilWidth()
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in range(xs, xe):
            ix = i-xs+stencil
            
            for j in range(ys, ye):
                jx = j-ys+stencil
                
                row.index = (i,j)
                
                for index, value in [
                    ((i,   j-1),                         - 1. * self.lapy_fac),
                    ((i-1, j  ),    - 1. * self.lapx_fac                ),
                    ((i,   j  ), 1. + 2. * self.lapx_fac + 2. * self.lapy_fac),
                    ((i+1, j  ),    - 1. * self.lapx_fac                ),
                    ((i,   j+1),                         - 1. * self.lapy_fac),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)

        A.assemble()
        

    def mult(self, Mat mat, Vec Q, Vec Y):
        self.matrix_mult(Q, Y)
        
    
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def matrix_mult(self, Vec Q, Vec Y):
        if self.de != 0.:
            self.derivatives.laplace_vec(Q, Y, -self.de**2)
        else:
            self.derivatives.arakawa_vec(self.Aa, Q, self.T)
            self.derivatives.arakawa_vec(self.Aa, self.T, self.Y)
            Y.scale(-0.5*self.ht*0.5*self.ht)

        Y.axpy(1., Q)
