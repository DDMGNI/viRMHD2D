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
        self.localL = self.da1.createLocalVec()
        
        # create local data and history vectors
        self.localXd = self.da4.createLocalVec()
        self.localXp = self.da4.createLocalVec()
        self.localXh = self.da4.createLocalVec()
        
        self.localAd = self.da1.createLocalVec()
        self.localJd = self.da1.createLocalVec()
        self.localPd = self.da1.createLocalVec()
        self.localOd = self.da1.createLocalVec()
        
        self.localQd = self.da1.createLocalVec()
        self.localT1 = self.da1.createLocalVec()
        self.localT2 = self.da1.createLocalVec()
        
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
#         self.poisson_ksp.setTolerances(rtol=1E-10, atol=1E-12)
        self.poisson_ksp.setTolerances(rtol=1E-10, atol=1E-12)
        self.poisson_ksp.setOperators(self.Pm)
#         self.poisson_ksp.setNullSpace(PETSc.NullSpace().create(constant=True))
#         self.poisson_ksp.setType('cg')
        self.poisson_ksp.setType('richardson')
#         self.poisson_ksp.setType('bcgs')
#         self.poisson_ksp.setType('gmres')
#         self.poisson_ksp.setType('fgmres')
#         self.poisson_ksp.getPC().setType('none')
#         self.poisson_ksp.getPC().setType('gamg')
#         self.poisson_ksp.getPC().setType('mg')    
#         self.poisson_ksp.getPC().setType('ml')
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
#         self.parabol_ksp.setTolerances(rtol=1E-8, atol=1E-14, max_it=1)
        self.parabol_ksp.setTolerances(rtol=1E-8, atol=1E-14)
#         self.parabol_ksp.setTolerances(max_it=1)
        self.parabol_ksp.setOperators(self.Qm)
        self.parabol_ksp.setType('cg')
#         self.parabol_ksp.setNormType(self.parabol_ksp.NormType.NORM_NONE)
        self.parabol_ksp.getPC().setType('none')
#         self.parabol_ksp.getPC().setType('hypre')
        
    
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
        
        cdef np.ndarray[double, ndim=3] f   = self.da4.getVecArray(self.F )[...]
        cdef np.ndarray[double, ndim=2] tfa = self.da1.getVecArray(self.FA)[...]
        cdef np.ndarray[double, ndim=2] tfj = self.da1.getVecArray(self.FJ)[...]
        cdef np.ndarray[double, ndim=2] tfp = self.da1.getVecArray(self.FP)[...]
        cdef np.ndarray[double, ndim=2] tfo = self.da1.getVecArray(self.FO)[...]
        
        tfa[:,:] = f[:,:,0]
        tfj[:,:] = f[:,:,1]
        tfp[:,:] = f[:,:,2]
        tfo[:,:] = f[:,:,3]
        
    
#     def compute_phi(self):
#         
#         self.L.copy(self.Pd)
#         
#         self.derivatives.arakawa_vec(self.Pp, self.Pd, self.T1)
#         self.derivatives.arakawa_vec(self.Ph, self.Pd, self.T2)
#         self.Pd.axpy(0.25*self.ht, self.T1)
#         self.Pd.axpy(0.25*self.ht, self.T2)
#         
#         self.derivatives.arakawa_vec(self.Ap, self.Ad, self.T1)
#         self.derivatives.arakawa_vec(self.Ah, self.Ad, self.T2)
#         self.Pd.axpy(0.25*self.ht, self.T1)
#         self.Pd.axpy(0.25*self.ht, self.T2)
#     
#     
#     def compute_psi(self):
#         
#         self.Qb.set(0.)
#         self.Qb.axpy(1., self.FA)
#         
#         self.derivatives.arakawa_vec(self.Ap, self.L, self.T1)
#         self.derivatives.arakawa_vec(self.Ah, self.L, self.T2)
#         self.Qb.axpy(0.25*self.ht, self.T1)
#         self.Qb.axpy(0.25*self.ht, self.T2)
#         
#         self.derivatives.arakawa_vec(self.Pp, self.Pd, self.T1)
#         self.derivatives.arakawa_vec(self.Ph, self.Pd, self.T2)
#         self.T.set(0.)
#         self.T.axpy(0.25*self.ht, self.T1)
#         self.T.axpy(0.25*self.ht, self.T2)
#         
#         self.derivatives.arakawa_vec(self.Ap, self.T, self.T1)
#         self.derivatives.arakawa_vec(self.Ah, self.T, self.T2)
#         self.Qb.axpy(0.25*self.ht, self.T1)
#         self.Qb.axpy(0.25*self.ht, self.T2)
#         
# #         print("    PC parabolic solve")
# #         self.Ad.set(0.)
#         self.parabol_ksp.solve(self.Qb, self.Ad)
    
    
    def solve(self, Vec X, Vec Y):
        cdef int i, j, k
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()        
        
        self.update_function(X)

        self.da4.globalToLocal(self.F,  self.localF)
        self.da4.globalToLocal(self.Xp, self.localXp)
        self.da4.globalToLocal(self.Xh, self.localXh)
        
        cdef np.ndarray[double, ndim=3] f  = self.da4.getVecArray(self.localF )[...]
        cdef np.ndarray[double, ndim=3] xp = self.da4.getVecArray(self.localXp)[...]
        cdef np.ndarray[double, ndim=3] xh = self.da4.getVecArray(self.localXh)[...]
        
        cdef double[:,:] fa = f[:,:,0]
        cdef double[:,:] fj = f[:,:,1]
        cdef double[:,:] fp = f[:,:,2]
        cdef double[:,:] fo = f[:,:,3]
        
        cdef double[:,:] A_ave = 0.5 * (xp[...][:,:,0] + xh[...][:,:,0])
        cdef double[:,:] J_ave = 0.5 * (xp[...][:,:,1] + xh[...][:,:,1])
        cdef double[:,:] P_ave = 0.5 * (xp[...][:,:,2] + xh[...][:,:,2])
        cdef double[:,:] O_ave = 0.5 * (xp[...][:,:,3] + xh[...][:,:,3])
        
        self.Ad.set(0.)
        self.Jd.set(0.)
        self.Pd.set(0.)
        self.Od.set(0.)
        
        cdef double[:,:] l
        cdef double[:,:] t
        cdef double[:,:] pb
        cdef double[:,:] qb
        cdef double[:,:] td
        cdef double[:,:] ad
        cdef double[:,:] pd
        
        pb = self.da1.getVecArray(self.Pb)[...]

        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
             
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                 
                pb[iy, jy] = fo[ix,jx] - fp[ix,jx] \
                           + 0.5 * self.ht * self.derivatives.arakawa(P_ave, fp, ix, jx) \
                           - 0.5 * self.ht * self.derivatives.arakawa(A_ave, fj, ix, jx)
        
        self.poisson_nullspace.remove(self.Pb)
        
        self.L.set(0.)
        self.poisson_ksp.solve(self.Pb, self.L)
        
        
        for k in range(4):
            
#             pb = self.da1.getVecArray(self.Pb)[...]
# 
#             self.da1.globalToLocal(self.Jd, self.localJd)
#             jd = self.da1.getVecArray(self.localJd)[...]
#             
#             self.da1.globalToLocal(self.Od, self.localOd)
#             od = self.da1.getVecArray(self.localOd)[...]
#             
#             for i in range(xs, xe):
#                 ix = i-xs+2
#                 iy = i-xs
#                  
#                 for j in range(ys, ye):
#                     jx = j-ys+2
#                     jy = j-ys
#                      
#                     pb[iy, jy] = fo[ix,jx] - fp[ix,jx] \
#                                + 0.5 * self.ht * self.derivatives.arakawa(P_ave, fp, ix, jx) \
#                                - 0.5 * self.ht * self.derivatives.arakawa(A_ave, fj, ix, jx) \
# #                                + 0.5 * self.ht * self.derivatives.arakawa(P_ave, od, ix, jx) \
# #                                - 0.5 * self.ht * self.derivatives.arakawa(A_ave, jd, ix, jx)
#             
#             self.L.set(0.)
#             self.poisson_ksp.solve(self.Pb, self.L)
            
            
            self.da1.globalToLocal(self.L,  self.localL)
            l = self.da1.getVecArray(self.localL)[...]
            
            
            
            self.Pd.copy(self.T2)
            self.da1.globalToLocal(self.T2, self.localT2)
            td = self.da1.getVecArray(self.localT2)[...]
            
            t = self.da1.getVecArray(self.T1)[...]
            
            for i in range(xs, xe):
                ix = i-xs+2
                iy = i-xs
                 
                for j in range(ys, ye):
                    jx = j-ys+2
                    jy = j-ys
                     
                    t[iy, jy] = 0.5 * self.ht * self.derivatives.arakawa(P_ave, td, ix, jx)
            
            self.da1.globalToLocal(self.T1, self.localT1)
            t = self.da1.getVecArray(self.localT1)[...]
    
            qb = self.da1.getVecArray(self.Qb)[...]
            
            self.da1.globalToLocal(self.Ad, self.localAd)
            ad = self.da1.getVecArray(self.localAd)[...]
            
            for i in range(xs, xe):
                ix = i-xs+2
                iy = i-xs
                 
                for j in range(ys, ye):
                    jx = j-ys+2
                    jy = j-ys
                     
                    qb[iy,jy] = fa[ix,jx] \
                              + 0.5 * self.ht * self.derivatives.arakawa(A_ave, l,  ix, jx) \
                              + 0.5 * self.ht * self.derivatives.arakawa(A_ave, t,  ix, jx) \
                              - 0.5 * self.ht * self.derivatives.arakawa(P_ave, ad, ix, jx)
            
            self.parabol_ksp.solve(self.Qb, self.Ad)
            
            
            self.da1.globalToLocal(self.Ad, self.localAd)
            ad = self.da1.getVecArray(self.localAd)[...]
            
            pd = self.da1.getVecArray(self.Pd)[...]
            
            for i in range(xs, xe):
                ix = i-xs+2
                iy = i-xs
                 
                for j in range(ys, ye):
                    jx = j-ys+2
                    jy = j-ys
                     
                    pd[iy,jy] = l[ix,jx] \
                              - 0.5 * self.ht * self.derivatives.arakawa(P_ave, td, ix, jx) \
                              + 0.5 * self.ht * self.derivatives.arakawa(A_ave, ad, ix, jx)
        
        
            self.da1.globalToLocal(self.Pd, self.localPd)
            pd = self.da1.getVecArray(self.localPd)[...]
            
            od = self.da1.getVecArray(self.Od)[...]
            jd = self.da1.getVecArray(self.Jd)[...]
            
            for i in range(xs, xe):
                ix = i-xs+2
                iy = i-xs
                 
                for j in range(ys, ye):
                    jx = j-ys+2
                    jy = j-ys
                    
                    od[iy,jy] = fp[ix,jx] - self.derivatives.laplace(pd, ix, jx)
                    jd[iy,jy] = fj[ix,jx] - self.derivatives.laplace(ad, ix, jx)
                
        
#         self.derivatives.laplace_vec(self.Pd, self.Od, -1.)
#         self.derivatives.laplace_vec(self.Ad, self.Jd, -1.)
#         
#         self.Od.axpy(1., self.FP)
#         self.Jd.axpy(1., self.FJ)
        
#         cdef double[:,:,:] tx = self.da4.getVecArray(Y)[...]
        cdef np.ndarray[double, ndim=3] tx = self.da4.getVecArray(Y)[...]
        ad = self.da1.getVecArray(self.Ad)[...]
        pd = self.da1.getVecArray(self.Pd)[...]
        
        tx[:,:,0] = ad[:,:]
        tx[:,:,1] = jd[:,:]
        tx[:,:,2] = pd[:,:]
        tx[:,:,3] = od[:,:]
    
    
    def mult(self, Mat mat, Vec Q, Vec Y):
        self.matrix_mult(Q, Y)
        
    
    @cython.boundscheck(False)
    def matrix_mult(self, Vec Q, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        self.da1.globalToLocal(Q,       self.localQd)
        self.da4.globalToLocal(self.Xp, self.localXp)
        self.da4.globalToLocal(self.Xh, self.localXh)
        
        cdef double[:,:] y  = self.da1.getVecArray(Y)[...]
        cdef double[:,:] qd = self.da1.getVecArray(self.localQd)[...]
        
        cdef np.ndarray[double, ndim=3] xp = self.da4.getVecArray(self.localXp)[...]
        cdef np.ndarray[double, ndim=3] xh = self.da4.getVecArray(self.localXh)[...]
          
        cdef double[:,:] T1 = self.da1.getVecArray(self.T1)[...]
#         cdef np.ndarray[double, ndim=2] T2 = self.da1.getVecArray(self.T2)[...]
        
        cdef double[:,:] A_ave = 0.5 * (xp[...][:,:,0] + xh[...][:,:,0])
        cdef double[:,:] P_ave = 0.5 * (xp[...][:,:,2] + xh[...][:,:,2])
        
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
#         Q.copy(Y)
#         
#         self.derivatives.arakawa_vec(self.Pp, Q, self.T1)
#         self.derivatives.arakawa_vec(self.Ph, Q, self.T2)
#         Y.axpy(0.25*self.ht, self.T1)
#         Y.axpy(0.25*self.ht, self.T2)
# 
#         self.derivatives.arakawa_vec(self.Ap, Q, self.T1)
#         self.derivatives.arakawa_vec(self.Ah, Q, self.T2)
#         self.T.set(0.)
#         self.T.axpy(0.25*self.ht, self.T1)
#         self.T.axpy(0.25*self.ht, self.T2)
#         
#         self.derivatives.arakawa_vec(self.Ap, self.T, self.T1)
#         self.derivatives.arakawa_vec(self.Ah, self.T, self.T2)
#         Y.axpy(-0.25*self.ht, self.T1)
#         Y.axpy(-0.25*self.ht, self.T2)
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
             
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                 
                T1[iy, jy] = 0.5 * self.ht * self.derivatives.arakawa(A_ave, qd, ix, jx)
        
        
        self.da1.globalToLocal(self.T1, self.localT1)
        T1 = self.da1.getVecArray(self.localT1)[...]
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
             
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                 
                y[iy, jy] = qd[ix,jx] \
                          - 0.5 * self.ht * self.derivatives.arakawa(A_ave, T1, ix, jx)
#                           + 0.5 * self.ht * self.derivatives.arakawa(P_ave, qd, ix, jx) \

