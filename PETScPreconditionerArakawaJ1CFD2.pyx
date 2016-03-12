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
        
        # create local solver vectors
        self.localF = self.da4.createLocalVec()
        self.localB = self.da4.createLocalVec()
        
        # create local data and history vectors
        self.localXd = self.da4.createLocalVec()
        self.localXp = self.da4.createLocalVec()
        self.localXh = self.da4.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETScDerivatives(da1, nx, ny, ht, hx, hy)
        
        # initialise rhs and matrix for Poisson solver
        self.Pb = self.da1.createGlobalVec()
        self.Pm = self.da1.createMat()
        self.Pm.setOption(self.Pm.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.Pm.setUp()
        self.Pm.setNullSpace(PETSc.NullSpace().create(constant=True))
        
        # create Poisson solver object and build matrix
        self.petsc_poisson = PETScPoisson(self.da1, self.nx, self.ny, self.hx, self.hy)
        self.petsc_poisson.formMat(self.Pm)
        
        # create linear Poisson solver
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setTolerances(rtol=1E-12, atol=1E-16)
        self.poisson_ksp.setOperators(self.Pm)
        self.poisson_ksp.setType('cg')
#         self.poisson_ksp.getPC().setType('none')
#         self.poisson_ksp.getPC().setType('hypre')
        self.poisson_ksp.getPC().setType('lu')
        self.poisson_ksp.getPC().setFactorSolverPackage('superlu_dist')

        # initialise rhs and matrixfree matrix for preconditioner
        self.Qb = self.da1.createGlobalVec()
        self.Qm = PETSc.Mat().createPython([self.Qb.getSizes(), self.Qb.getSizes()], 
                                            context=self,
                                            comm=PETSc.COMM_WORLD)
        self.Qm.setUp()

        # create linear parabolic solver
        self.parabol_ksp = PETSc.KSP().create()
        self.parabol_ksp.setFromOptions()
        self.parabol_ksp.setTolerances(rtol=1E-8, atol=1E-14)
        self.parabol_ksp.setOperators(self.Qm)
        self.parabol_ksp.setType('gmres')
        self.parabol_ksp.getPC().setType('none')
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
        x = self.da4.getVecArray(self.Xh)
        a = self.da1.getVecArray(self.Ah)
        j = self.da1.getVecArray(self.Jh)
        p = self.da1.getVecArray(self.Ph)
        o = self.da1.getVecArray(self.Oh)
        
        a[:,:] = x[:,:,0]
        j[:,:] = x[:,:,1]
        p[:,:] = x[:,:,2]
        o[:,:] = x[:,:,3]
        
    
    def update_previous(self, Vec X):
        X.copy(self.Xp)
        
        x = self.da4.getVecArray(self.Xp)
        a = self.da1.getVecArray(self.Ap)
        j = self.da1.getVecArray(self.Jp)
        p = self.da1.getVecArray(self.Pp)
        o = self.da1.getVecArray(self.Op)
        
        a[:,:] = x[:,:,0]
        j[:,:] = x[:,:,1]
        p[:,:] = x[:,:,2]
        o[:,:] = x[:,:,3]
        
    
    def update_function(self, Vec F):
        F.copy(self.F)
        self.F.scale(-1.)
        
        f  = self.da4.getVecArray(self.F)
        fa = self.da1.getVecArray(self.FA)
        fj = self.da1.getVecArray(self.FJ)
        fp = self.da1.getVecArray(self.FP)
        fo = self.da1.getVecArray(self.FO)
        
        fa[:,:] = f[:,:,0]
        fj[:,:] = f[:,:,1]
        fp[:,:] = f[:,:,2]
        fo[:,:] = f[:,:,3]
        
        self.Pb.set(0.)
        self.Pb.axpy(+1., self.FO)
        self.Pb.axpy(-1., self.FP)
        
        self.derivatives.arakawa_vec(self.Pp, self.FP, self.T1)
        self.derivatives.arakawa_vec(self.Ph, self.FP, self.T2)
        self.Pb.axpy(-0.25*self.ht, self.T1)
        self.Pb.axpy(-0.25*self.ht, self.T2)
        
        self.derivatives.arakawa_vec(self.Ap, self.FJ, self.T1)
        self.derivatives.arakawa_vec(self.Ah, self.FJ, self.T2)
        self.Pb.axpy(+0.25*self.ht, self.T1)
        self.Pb.axpy(+0.25*self.ht, self.T2)
        
#         print("    PC poisson solve")
        self.L.set(0.)
        self.poisson_ksp.solve(self.Pb, self.L)
        
    
    def compute_phi(self):
        
        self.L.copy(self.Pd)
        
        self.derivatives.arakawa_vec(self.Pp, self.Pd, self.T1)
        self.derivatives.arakawa_vec(self.Ph, self.Pd, self.T2)
        self.Pd.axpy(0.25*self.ht, self.T1)
        self.Pd.axpy(0.25*self.ht, self.T2)
        
        self.derivatives.arakawa_vec(self.Ap, self.Ad, self.T1)
        self.derivatives.arakawa_vec(self.Ah, self.Ad, self.T2)
        self.Pd.axpy(0.25*self.ht, self.T1)
        self.Pd.axpy(0.25*self.ht, self.T2)
    
    
    def compute_psi(self):
        
        self.Qb.set(0.)
        self.Qb.axpy(-1., self.FA)
        
        self.derivatives.arakawa_vec(self.Ap, self.L, self.T1)
        self.derivatives.arakawa_vec(self.Ah, self.L, self.T2)
        self.Qb.axpy(0.25*self.ht, self.T1)
        self.Qb.axpy(0.25*self.ht, self.T2)
        
        self.derivatives.arakawa_vec(self.Pp, self.Pd, self.T1)
        self.derivatives.arakawa_vec(self.Ph, self.Pd, self.T2)
        self.T.set(0.)
        self.T.axpy(0.25*self.ht, self.T1)
        self.T.axpy(0.25*self.ht, self.T2)
        
        self.derivatives.arakawa_vec(self.Ap, self.T, self.T1)
        self.derivatives.arakawa_vec(self.Ah, self.T, self.T2)
        self.Qb.axpy(0.25*self.ht, self.T1)
        self.Qb.axpy(0.25*self.ht, self.T2)
        
#         print("    PC parabolic solve")
#         self.Ad.set(0.)
        self.parabol_ksp.solve(self.Qb, self.Ad)
    
    
    def solve(self, Vec X, Vec Y):
        
        self.update_function(X)
        
        self.Ad.set(0.)
        
        for i in range(4):
            self.compute_psi()
            self.compute_phi()
        
        self.derivatives.laplace_vec(self.Pd, self.Od, -1.)
        self.derivatives.laplace_vec(self.Ad, self.Jd, -1.)
        
        self.Od.axpy(-1., self.FP)
        self.Jd.axpy(-1., self.FJ)
        
        x = self.da4.getVecArray(self.Xd)
        a = self.da1.getVecArray(self.Ad)
        j = self.da1.getVecArray(self.Jd)
        p = self.da1.getVecArray(self.Pd)
        o = self.da1.getVecArray(self.Od)
        
        x[:,:,0] = a[:,:]
        x[:,:,1] = j[:,:]
        x[:,:,2] = p[:,:]
        x[:,:,3] = o[:,:]
        
        self.Xd.copy(Y)
    
    
    def mult(self, Mat mat, Vec Q, Vec Y):
        self.matrix_mult(Q, Y)
        
    
    @cython.boundscheck(False)
    def matrix_mult(self, Vec Q, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        Q.copy(Y)
        
        self.derivatives.arakawa_vec(self.Pp, Q, self.T1)
        self.derivatives.arakawa_vec(self.Ph, Q, self.T2)
        Y.axpy(0.25*self.ht, self.T1)
        Y.axpy(0.25*self.ht, self.T2)

        self.derivatives.arakawa_vec(self.Ap, Q, self.T1)
        self.derivatives.arakawa_vec(self.Ah, Q, self.T2)
        self.T.set(0.)
        self.T.axpy(0.25*self.ht, self.T1)
        self.T.axpy(0.25*self.ht, self.T2)
        
        self.derivatives.arakawa_vec(self.Ap, self.T, self.T1)
        self.derivatives.arakawa_vec(self.Ah, self.T, self.T2)
        Y.axpy(-0.25*self.ht, self.T1)
        Y.axpy(-0.25*self.ht, self.T2)
