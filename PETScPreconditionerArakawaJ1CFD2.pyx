# cython: profile=True
'''
Created on Apr 10, 2012

@author: mkraus
'''
from numpy import ix_

cimport cython

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
                 double ht, double hx, double hy):
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
        
        # create solver vectors
        self.L  = self.da1.createGlobalVec()
        self.F  = self.da4.createGlobalVec()
        
        self.FA = self.da1.createGlobalVec()
        self.FJ = self.da1.createGlobalVec()
        self.FP = self.da1.createGlobalVec()
        self.FO = self.da1.createGlobalVec()
        
        self.T  = self.da1.createGlobalVec()
        self.T1 = self.da1.createGlobalVec()
        self.T2 = self.da1.createGlobalVec()
        self.T3 = self.da1.createGlobalVec()
        self.T4 = self.da1.createGlobalVec()
        
        # create data and history vectors
        self.Xd = self.da4.createGlobalVec()
        self.Xp = self.da4.createGlobalVec()
        self.Xh = self.da4.createGlobalVec()
        
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
        self.poisson_ksp.setTolerances(rtol=1E-10, atol=1E-12)
        self.poisson_ksp.setType('cg')
        self.poisson_ksp.getPC().setType('hypre')
#         self.poisson_ksp.getPC().setType('lu')
#         self.poisson_ksp.getPC().setFactorSolverPackage('superlu_dist')

        # initialise rhs and matrixfree matrix for preconditioner
        self.Qb = self.da1.createGlobalVec()
        self.Qm = PETSc.Mat().createPython([self.Qb.getSizes(), self.Qb.getSizes()], 
                                            context=self,
                                            comm=PETSc.COMM_WORLD)
        self.Qm.setUp()

        # create linear parabolic solver
        self.parabol_ksp = PETSc.KSP().create()
        self.parabol_ksp.setFromOptions()
        self.parabol_ksp.setOperators(self.Qm)
#         self.parabol_ksp.setTolerances(rtol=1E-8, atol=1E-14, max_it=1)
        self.parabol_ksp.setTolerances(rtol=1E-8, atol=1E-14)
#         self.parabol_ksp.setTolerances(max_it=1)
#         self.parabol_ksp.setNormType(self.parabol_ksp.NormType.NORM_NONE)
        self.parabol_ksp.setType('cg')
        self.parabol_ksp.getPC().setType('none')
        
    
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
        
    
    def update_function(self, Vec F):
        F.copy(self.F)
        
        f = self.da4.getVecArray(self.F)
        
        self.da1.getVecArray(self.FA)[:,:] = f[:,:,0]
        self.da1.getVecArray(self.FJ)[:,:] = f[:,:,1]
        self.da1.getVecArray(self.FP)[:,:] = f[:,:,2]
        self.da1.getVecArray(self.FO)[:,:] = f[:,:,3]
        
    
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
        self.Pd.set(0.)
        
        for k in range(4):
            
            self.derivatives.arakawa_vec(self.Pa, self.Pd, self.T)
            self.derivatives.arakawa_vec(self.Aa, self.L,  self.T1)
            self.derivatives.arakawa_vec(self.Aa, self.T,  self.T2)
            self.derivatives.arakawa_vec(self.Pa, self.Ad, self.T3)
            self.derivatives.arakawa_vec(self.Aa, self.Ad, self.T4)
            
            self.da1.getVecArray(self.Qb)[:,:] = self.da1.getVecArray(self.FA)[:,:] \
                                               + 0.5*self.ht * self.da1.getVecArray(self.T1)[:,:] \
                                               + 0.5*self.ht * 0.5*self.ht * self.da1.getVecArray(self.T2)[:,:] \
                                               - 0.5*self.ht * self.da1.getVecArray(self.T3)[:,:]

            self.parabol_ksp.solve(self.Qb, self.Ad)
            
            
            self.da1.getVecArray(self.Pd)[:,:] = self.da1.getVecArray(self.L)[:,:] \
                                               - 0.5*self.ht * self.da1.getVecArray(self.T )[:,:] \
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
    
    
    def mult(self, Mat mat, Vec Q, Vec Y):
        self.matrix_mult(Q, Y)
        
    
    def matrix_mult(self, Vec Q, Vec Y):
        self.derivatives.arakawa_vec(self.Aa, Q, self.T1)
        self.derivatives.arakawa_vec(self.Aa, self.T1, self.T2)
        Y.waxpy(-0.5*self.ht*0.5*self.ht, self.T2, Q)
#         self.da1.getVecArray(Y)[:,:] = self.da1.getVecArray(Q)[:,:] - 0.5*self.ht*0.5*self.ht * self.da1.getVecArray(self.T2)[:,:]   
