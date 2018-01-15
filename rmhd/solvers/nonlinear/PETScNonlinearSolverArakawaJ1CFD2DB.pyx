'''
Created on Apr 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport Mat, Vec

from PETScNonlinearSolverArakawaJ1CFD2 cimport PETScSolver


cdef class PETScSolverDB(PETScSolver):
    '''
    The PETScSolver class implements a nonlinear solver for the reduced MHD system
    built on top of the PETSc SNES module.
    '''
    
    @cython.boundscheck(False)
    def mult(self, Mat mat, Vec X, Vec Y):
        
        cdef double sign = -1.
        
        super().mult(mat, X, Y)
        

        y = self.da4.getVecArray(Y)
         
        # magnetic potential
        # [A, [O, P]]
        self.derivatives.arakawa_vec(self.Od, self.Pa, self.T1)
        self.derivatives.arakawa_vec(self.Aa, self.T1, self.T2)
        
        y[:,:,0] += sign * 0.5 * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        self.derivatives.arakawa_vec(self.Oa, self.Pd, self.T1)
        self.derivatives.arakawa_vec(self.Aa, self.T1, self.T2)
        
        y[:,:,0] += sign * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        self.derivatives.arakawa_vec(self.Oa, self.Pa, self.T1)
        self.derivatives.arakawa_vec(self.Ad, self.T1, self.T2)
        
        y[:,:,0] += sign * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        
        # [A, [A, J]]
        self.derivatives.arakawa_vec(self.Ad, self.Ja, self.T1)
        self.derivatives.arakawa_vec(self.Aa, self.T1, self.T2)
        
        y[:,:,0] += sign * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]

        self.derivatives.arakawa_vec(self.Aa, self.Jd, self.T1)
        self.derivatives.arakawa_vec(self.Aa, self.T1, self.T2)
        
        y[:,:,0] += sign * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]

        self.derivatives.arakawa_vec(self.Aa, self.Ja, self.T1)
        self.derivatives.arakawa_vec(self.Ad, self.T1, self.T2)
        
        y[:,:,0] += sign * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        
        # vorticity
        # [O, [O, P]]
        self.derivatives.arakawa_vec(self.Od, self.Pa, self.T1)
        self.derivatives.arakawa_vec(self.Oa, self.T1, self.T2)
        
        y[:,:,3] += sign * 0.5 * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        self.derivatives.arakawa_vec(self.Oa, self.Pd, self.T1)
        self.derivatives.arakawa_vec(self.Oa, self.T1, self.T2)
        
        y[:,:,3] += sign * 0.5 * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        self.derivatives.arakawa_vec(self.Oa, self.Pa, self.T1)
        self.derivatives.arakawa_vec(self.Od, self.T1, self.T2)
        
        y[:,:,3] += sign * 0.5 * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        
        # [O, [A, J]]
        self.derivatives.arakawa_vec(self.Ad, self.Ja, self.T1)
        self.derivatives.arakawa_vec(self.Oa, self.T1, self.T2)
        
        y[:,:,3] += sign * 0.5 * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        self.derivatives.arakawa_vec(self.Aa, self.Jd, self.T1)
        self.derivatives.arakawa_vec(self.Oa, self.T1, self.T2)
        
        y[:,:,3] += sign * 0.5 * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        self.derivatives.arakawa_vec(self.Aa, self.Ja, self.T1)
        self.derivatives.arakawa_vec(self.Od, self.T1, self.T2)
        
        y[:,:,3] += sign * 0.5 * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        
        # [A, [A, P]]
        self.derivatives.arakawa_vec(self.Ad, self.Pa, self.T1)
        self.derivatives.arakawa_vec(self.Aa, self.T1, self.T2)
        
        y[:,:,3] += sign * 0.5 * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        self.derivatives.arakawa_vec(self.Aa, self.Pd, self.T1)
        self.derivatives.arakawa_vec(self.Aa, self.T1, self.T2)
        
        y[:,:,3] += sign * 0.5 * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        self.derivatives.arakawa_vec(self.Aa, self.Pa, self.T1)
        self.derivatives.arakawa_vec(self.Ad, self.T1, self.T2)
        
        y[:,:,3] += sign * 0.5 * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]


   
    @cython.boundscheck(False)
    def function(self, Vec Y):
        
        cdef double sign = -1.
        
        super().function(Y)
        
        
        y = self.da4.getVecArray(Y)
         
        # magnetic potential
        # [A, [O, P]]
        self.derivatives.arakawa_vec(self.Oa, self.Pa, self.T1)
        self.derivatives.arakawa_vec(self.Aa, self.T1, self.T2)
        
        y[:,:,0] += sign * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        # [A, [A, J]]
        self.derivatives.arakawa_vec(self.Aa, self.Ja, self.T1)
        self.derivatives.arakawa_vec(self.Aa, self.T1, self.T2)
        
        y[:,:,0] += sign * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        
        # vorticity
        # [O, [O, P]]
        self.derivatives.arakawa_vec(self.Oa, self.Pa, self.T1)
        self.derivatives.arakawa_vec(self.Oa, self.T1, self.T2)
        
        y[:,:,3] += sign * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        # [O, [A, J]]
        self.derivatives.arakawa_vec(self.Aa, self.Ja, self.T1)
        self.derivatives.arakawa_vec(self.Oa, self.T1, self.T2)
        
        y[:,:,3] += sign * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
        # [A, [A, P]]
        self.derivatives.arakawa_vec(self.Aa, self.Pa, self.T1)
        self.derivatives.arakawa_vec(self.Aa, self.T1, self.T2)
        
        y[:,:,3] += sign * self.nu * self.ht * self.da1.getVecArray(self.T2)[:,:]
        
