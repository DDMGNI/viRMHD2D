'''
Created on Apr 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport Mat, Vec

from rmhd.solvers.common.PETScDerivatives import PETScDerivatives


cdef class PETScSolverDOF2(object):
    '''
    The PETScSolver class implements a nonlinear solver for the reduced MHD system
    built on top of the PETSc SNES module.
    '''
    
    def __init__(self, object da1, object da2,
                 int nx, int ny,
                 double ht, double hx, double hy,
                 object pc=None):
        '''
        Constructor
        '''
        
        # distributed arrays
        self.da1 = da1
        self.da2 = da2
        
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
        self.Yd = self.da2.createGlobalVec()
        
        self.FA = self.da1.createGlobalVec()
        self.FJ = self.da1.createGlobalVec()
        self.FP = self.da1.createGlobalVec()
        self.FO = self.da1.createGlobalVec()
        
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
        
        
    
    def update_history(self):
        self.Ap.copy(self.Ah)
        self.Jp.copy(self.Jh)
        self.Pp.copy(self.Ph)
        self.Op.copy(self.Oh)
        
        if self.pc is not None:
            self.pc.update_history(self.Ah, self.Jh, self.Ph, self.Oh)
        
        
    
    def update_previous(self, Vec Ap, Vec Jp, Vec Pp, Vec Op):
        Ap.copy(self.Ap)
        Jp.copy(self.Jp)
        Pp.copy(self.Pp)
        Op.copy(self.Op)
        
        self.da1.getVecArray(self.Aa)[:,:] = 0.5 * (self.da1.getVecArray(self.Ap)[:,:] + self.da1.getVecArray(self.Ah)[:,:])
        self.da1.getVecArray(self.Ja)[:,:] = 0.5 * (self.da1.getVecArray(self.Jp)[:,:] + self.da1.getVecArray(self.Jh)[:,:])
        self.da1.getVecArray(self.Pa)[:,:] = 0.5 * (self.da1.getVecArray(self.Pp)[:,:] + self.da1.getVecArray(self.Ph)[:,:])
        self.da1.getVecArray(self.Oa)[:,:] = 0.5 * (self.da1.getVecArray(self.Op)[:,:] + self.da1.getVecArray(self.Oh)[:,:])
        
        self.da1.globalToLocal(self.Aa, self.localAa)
        self.da1.globalToLocal(self.Ja, self.localJa)
        self.da1.globalToLocal(self.Pa, self.localPa)
        self.da1.globalToLocal(self.Oa, self.localOa)
        
        if self.pc is not None:
            self.pc.update_previous(self.Ap, self.Jp, self.Pp, self.Op)
    
    
    def update_function(self, Vec FA, Vec FJ, Vec FP, Vec FO):
        FA.copy(self.FA)
        FJ.copy(self.FJ)
        FP.copy(self.FP)
        FO.copy(self.FO)
        
        
    @cython.boundscheck(False)
    def mult(self, Mat mat, Vec X, Vec Y):
        
        if self.pc == None:
            X.copy(self.Yd)
        else:
            self.pc.solve(X, self.Yd)
        
        xd = self.da2.getVecArray(self.Yd)
        
        self.da1.getVecArray(self.Ad)[:,:] = xd[:,:,0]
        self.da1.getVecArray(self.Pd)[:,:] = xd[:,:,1]
        
        
        self.derivatives.laplace_vec(self.Ad, self.Jd, -1.)
        self.derivatives.laplace_vec(self.Pd, self.Od, -1.)
        
        self.Jd.axpy(-1., self.FJ)
        self.Od.axpy(-1., self.FP)

        y = self.da2.getVecArray(Y)
         
        # magnetic potential
        self.derivatives.arakawa_vec(self.Pa, self.Ad, self.T1)
        self.derivatives.arakawa_vec(self.Pd, self.Aa, self.T2)
        
        y[:,:,0] = self.da1.getVecArray(self.Ad)[:,:] \
                 + 0.5*self.ht * self.da1.getVecArray(self.T1)[:,:] \
                 + 0.5*self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        # vorticity
        self.derivatives.arakawa_vec(self.Pa, self.Od, self.T1)
        self.derivatives.arakawa_vec(self.Pd, self.Oa, self.T2)
        self.derivatives.arakawa_vec(self.Ja, self.Ad, self.T3)
        self.derivatives.arakawa_vec(self.Jd, self.Aa, self.T4)
        
        y[:,:,1] = self.da1.getVecArray(self.Od)[:,:] \
                 + 0.5*self.ht * self.da1.getVecArray(self.T1)[:,:] \
                 + 0.5*self.ht * self.da1.getVecArray(self.T2)[:,:] \
                 + 0.5*self.ht * self.da1.getVecArray(self.T3)[:,:] \
                 + 0.5*self.ht * self.da1.getVecArray(self.T4)[:,:]

