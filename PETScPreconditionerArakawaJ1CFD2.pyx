'''
Created on Apr 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

from datetime import datetime

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport PC, Mat, Vec

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
        
        self.arakawa_fac  = 0.5 * self.ht * self.hx_inv * self.hy_inv / 12.
        self.arakawa_fac2 = self.arakawa_fac**2 
        
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
        
        # create data and history vectors
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
        self.localL  = self.da1.createLocalVec()
        self.localQ  = self.da1.createLocalVec()
        self.localT  = self.da1.createLocalVec()
        
        self.localFA = self.da1.createLocalVec()
        self.localFJ = self.da1.createLocalVec()
        self.localFP = self.da1.createLocalVec()
        self.localFO = self.da1.createLocalVec()
        
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
        self.Pm.setOption(PETSc.Option.NEW_NONZERO_ALLOCATION_ERR, False)
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
        self.parabol_ksp.setOperators(self.Qm)
        self.parabol_ksp.setType('cg')
        self.parabol_ksp.getPC().setType('none')
#         self.parabol_ksp.setNormType(PETSc.NormType.NORM_NONE)
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
        
        self.da1.globalToLocal(self.Aa, self.localAa)
        self.da1.globalToLocal(self.Pa, self.localPa)
        
        self.aa = self.da1.getVecArray(self.localAa)[...]
        self.pa = self.da1.getVecArray(self.localPa)[...]
        
    
    def update_function(self, Vec F):
        F.copy(self.F)
        
        f = self.da4.getVecArray(self.F)
        
        self.da1.getVecArray(self.FA)[:,:] = f[:,:,0]
        self.da1.getVecArray(self.FJ)[:,:] = f[:,:,1]
        self.da1.getVecArray(self.FP)[:,:] = f[:,:,2]
        self.da1.getVecArray(self.FO)[:,:] = f[:,:,3]
        
        self.da1.globalToLocal(self.FA, self.localFA)
        self.da1.globalToLocal(self.FJ, self.localFJ)
        self.da1.globalToLocal(self.FP, self.localFP)
        self.da1.globalToLocal(self.FO, self.localFO)
        
        self.fa = self.da1.getVecArray(self.localFA)[...]
        self.fj = self.da1.getVecArray(self.localFJ)[...]
        self.fp = self.da1.getVecArray(self.localFP)[...]
        self.fo = self.da1.getVecArray(self.localFO)[...]
        
    
    def apply(self, PC pc, Vec F, Vec Y):
        self.solve(F, Y)
        
        
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def solve(self, Vec F, Vec Y):
        
        self.update_function(F)

        cdef int i, j, stencil
        cdef int xs, xe, ys, ye
        cdef double jpp, jpc, jcp, result
 
        (xs, xe), (ys, ye) = self.da1.getRanges()
        stencil = self.da1.getStencilWidth()
        
        
        cdef double[:,:] aa = self.aa
        cdef double[:,:] pa = self.pa
         
        cdef double[:,:] fa = self.fa
        cdef double[:,:] fj = self.fj
        cdef double[:,:] fp = self.fp
        cdef double[:,:] fo = self.fo
        
        cdef double[:,:] pb = self.da1.getVecArray(self.Pb)[...]
        cdef double[:,:] qb

        cdef double[:,:] l
        cdef double[:,:] t
        cdef double[:,:] ad
        cdef double[:,:] pd
        
        
        for i in range(stencil, xe-xs+stencil):
            for j in range(stencil, ye-ys+stencil):
                
                result = fo[i, j] - fp[i, j]
                
                jpp = (pa[i+1, j  ] - pa[i-1, j  ]) * (fp[i,   j+1] - fp[i,   j-1]) \
                    - (pa[i,   j+1] - pa[i,   j-1]) * (fp[i+1, j  ] - fp[i-1, j  ])
                 
                jpc = pa[i+1, j  ] * (fp[i+1, j+1] - fp[i+1, j-1]) \
                    - pa[i-1, j  ] * (fp[i-1, j+1] - fp[i-1, j-1]) \
                    - pa[i,   j+1] * (fp[i+1, j+1] - fp[i-1, j+1]) \
                    + pa[i,   j-1] * (fp[i+1, j-1] - fp[i-1, j-1])
                 
                jcp = pa[i+1, j+1] * (fp[i,   j+1] - fp[i+1, j  ]) \
                    - pa[i-1, j-1] * (fp[i-1, j  ] - fp[i,   j-1]) \
                    - pa[i-1, j+1] * (fp[i,   j+1] - fp[i-1, j  ]) \
                    + pa[i+1, j-1] * (fp[i+1, j  ] - fp[i,   j-1])
                 
                result += self.arakawa_fac * (jpp + jpc + jcp)
         
                 
                jpp = (aa[i+1, j  ] - aa[i-1, j  ]) * (fj[i,   j+1] - fj[i,   j-1]) \
                    - (aa[i,   j+1] - aa[i,   j-1]) * (fj[i+1, j  ] - fj[i-1, j  ])
                 
                jpc = aa[i+1, j  ] * (fj[i+1, j+1] - fj[i+1, j-1]) \
                    - aa[i-1, j  ] * (fj[i-1, j+1] - fj[i-1, j-1]) \
                    - aa[i,   j+1] * (fj[i+1, j+1] - fj[i-1, j+1]) \
                    + aa[i,   j-1] * (fj[i+1, j-1] - fj[i-1, j-1])
                 
                jcp = aa[i+1, j+1] * (fj[i,   j+1] - fj[i+1, j  ]) \
                    - aa[i-1, j-1] * (fj[i-1, j  ] - fj[i,   j-1]) \
                    - aa[i-1, j+1] * (fj[i,   j+1] - fj[i-1, j  ]) \
                    + aa[i+1, j-1] * (fj[i+1, j  ] - fj[i,   j-1])
                 
                result -= self.arakawa_fac * (jpp + jpc + jcp)
                
                pb[i-stencil, j-stencil] = result
                        
        
#         self.L.set(0.)
        self.poisson_nullspace.remove(self.Pb)
        self.poisson_ksp.solve(self.Pb, self.L)
        
        
        self.Ad.set(0.)
        self.Pd.set(0.)
        
        self.da1.globalToLocal(self.L,  self.localL )
        self.da1.globalToLocal(self.Ad, self.localAd)
        self.da1.globalToLocal(self.Pd, self.localPd)
         
        l  = self.da1.getVecArray(self.localL )[...]
        ad = self.da1.getVecArray(self.localAd)[...]
        pd = self.da1.getVecArray(self.localPd)[...]
         
        for k in range(self.jacobi_max_it):
            
            if k == 0:
                self.T.set(0)
            else:
                t = self.da1.getVecArray(self.T)[...]
                
                for i in range(stencil, xe-xs+stencil):
                    for j in range(stencil, ye-ys+stencil):
                        jpp = (pa[i+1, j  ] - pa[i-1, j  ]) * (pd[i,   j+1] - pd[i,   j-1]) \
                            - (pa[i,   j+1] - pa[i,   j-1]) * (pd[i+1, j  ] - pd[i-1, j  ])
                         
                        jpc = pa[i+1, j  ] * (pd[i+1, j+1] - pd[i+1, j-1]) \
                            - pa[i-1, j  ] * (pd[i-1, j+1] - pd[i-1, j-1]) \
                            - pa[i,   j+1] * (pd[i+1, j+1] - pd[i-1, j+1]) \
                            + pa[i,   j-1] * (pd[i+1, j-1] - pd[i-1, j-1])
                         
                        jcp = pa[i+1, j+1] * (pd[i,   j+1] - pd[i+1, j  ]) \
                            - pa[i-1, j-1] * (pd[i-1, j  ] - pd[i,   j-1]) \
                            - pa[i-1, j+1] * (pd[i,   j+1] - pd[i-1, j  ]) \
                            + pa[i+1, j-1] * (pd[i+1, j  ] - pd[i,   j-1])
                         
                        t[i-stencil, j-stencil] = (jpp + jpc + jcp)
            
            
            self.da1.globalToLocal(self.T, self.localT)
            t  = self.da1.getVecArray(self.localT)[...]
            qb = self.da1.getVecArray(self.Qb)[...]
         
            for i in range(stencil, xe-xs+stencil):
                for j in range(stencil, ye-ys+stencil):
                    
                    result = fa[i, j]
                    
                    jpp = (aa[i+1, j  ] - aa[i-1, j  ]) * (l[i,   j+1] - l[i,   j-1]) \
                        - (aa[i,   j+1] - aa[i,   j-1]) * (l[i+1, j  ] - l[i-1, j  ])
                     
                    jpc = aa[i+1, j  ] * (l[i+1, j+1] - l[i+1, j-1]) \
                        - aa[i-1, j  ] * (l[i-1, j+1] - l[i-1, j-1]) \
                        - aa[i,   j+1] * (l[i+1, j+1] - l[i-1, j+1]) \
                        + aa[i,   j-1] * (l[i+1, j-1] - l[i-1, j-1])
                     
                    jcp = aa[i+1, j+1] * (l[i,   j+1] - l[i+1, j  ]) \
                        - aa[i-1, j-1] * (l[i-1, j  ] - l[i,   j-1]) \
                        - aa[i-1, j+1] * (l[i,   j+1] - l[i-1, j  ]) \
                        + aa[i+1, j-1] * (l[i+1, j  ] - l[i,   j-1])
                     
                    result += self.arakawa_fac * (jpp + jpc + jcp)
                    
                    
                    jpp = (pa[i+1, j  ] - pa[i-1, j  ]) * (ad[i,   j+1] - ad[i,   j-1]) \
                        - (pa[i,   j+1] - pa[i,   j-1]) * (ad[i+1, j  ] - ad[i-1, j  ])
                     
                    jpc = pa[i+1, j  ] * (ad[i+1, j+1] - ad[i+1, j-1]) \
                        - pa[i-1, j  ] * (ad[i-1, j+1] - ad[i-1, j-1]) \
                        - pa[i,   j+1] * (ad[i+1, j+1] - ad[i-1, j+1]) \
                        + pa[i,   j-1] * (ad[i+1, j-1] - ad[i-1, j-1])
                     
                    jcp = pa[i+1, j+1] * (ad[i,   j+1] - ad[i+1, j  ]) \
                        - pa[i-1, j-1] * (ad[i-1, j  ] - ad[i,   j-1]) \
                        - pa[i-1, j+1] * (ad[i,   j+1] - ad[i-1, j  ]) \
                        + pa[i+1, j-1] * (ad[i+1, j  ] - ad[i,   j-1])
                     
                    result -= self.arakawa_fac * (jpp + jpc + jcp)
                    
             
                    jpp = (aa[i+1, j  ] - aa[i-1, j  ]) * (t[i,   j+1] - t[i,   j-1]) \
                        - (aa[i,   j+1] - aa[i,   j-1]) * (t[i+1, j  ] - t[i-1, j  ])
                     
                    jpc = aa[i+1, j  ] * (t[i+1, j+1] - t[i+1, j-1]) \
                        - aa[i-1, j  ] * (t[i-1, j+1] - t[i-1, j-1]) \
                        - aa[i,   j+1] * (t[i+1, j+1] - t[i-1, j+1]) \
                        + aa[i,   j-1] * (t[i+1, j-1] - t[i-1, j-1])
                     
                    jcp = aa[i+1, j+1] * (t[i,   j+1] - t[i+1, j  ]) \
                        - aa[i-1, j-1] * (t[i-1, j  ] - t[i,   j-1]) \
                        - aa[i-1, j+1] * (t[i,   j+1] - t[i-1, j  ]) \
                        + aa[i+1, j-1] * (t[i+1, j  ] - t[i,   j-1])
                     
                    result -= self.arakawa_fac2 * (jpp + jpc + jcp)
                    
                    qb[i-stencil, j-stencil] = result
                    
            
#             self.Ad.set(0.)
            self.parabol_ksp.solve(self.Qb, self.Ad)
            
            
            self.da1.globalToLocal(self.Ad, self.localAd)
            ad = self.da1.getVecArray(self.localAd)[...]
            pd = self.da1.getVecArray(self.Pd)[...]
            
            for i in range(stencil, xe-xs+stencil):
                for j in range(stencil, ye-ys+stencil):
                    
                    jpp = (aa[i+1, j  ] - aa[i-1, j  ]) * (ad[i,   j+1] - ad[i,   j-1]) \
                        - (aa[i,   j+1] - aa[i,   j-1]) * (ad[i+1, j  ] - ad[i-1, j  ])
                     
                    jpc = aa[i+1, j  ] * (ad[i+1, j+1] - ad[i+1, j-1]) \
                        - aa[i-1, j  ] * (ad[i-1, j+1] - ad[i-1, j-1]) \
                        - aa[i,   j+1] * (ad[i+1, j+1] - ad[i-1, j+1]) \
                        + aa[i,   j-1] * (ad[i+1, j-1] - ad[i-1, j-1])
                     
                    jcp = aa[i+1, j+1] * (ad[i,   j+1] - ad[i+1, j  ]) \
                        - aa[i-1, j-1] * (ad[i-1, j  ] - ad[i,   j-1]) \
                        - aa[i-1, j+1] * (ad[i,   j+1] - ad[i-1, j  ]) \
                        + aa[i+1, j-1] * (ad[i+1, j  ] - ad[i,   j-1])
                     
                    pd[i-stencil, j-stencil] = l[i, j] \
                               - self.arakawa_fac * t[i, j] \
                               + self.arakawa_fac * (jpp + jpc + jcp)
                    
            self.da1.globalToLocal(self.Pd, self.localPd)
            pd = self.da1.getVecArray(self.localPd)[...]
            
        
        cdef double[:,:,:] y = self.da4.getVecArray(Y)[...]
        
        for i in range(stencil, xe-xs+stencil):
            for j in range(stencil, ye-ys+stencil):
                
                y[i-stencil, j-stencil, 0] = ad[i, j]
                
                y[i-stencil, j-stencil, 1] = fj[i, j] \
                             + ( \
                                   - 1. * ad[i-1, j] \
                                   + 2. * ad[i,   j] \
                                   - 1. * ad[i+1, j] \
                               ) * self.hx_inv**2 \
                             + ( \
                                   - 1. * ad[i, j-1] \
                                   + 2. * ad[i, j  ] \
                                   - 1. * ad[i, j+1] \
                               ) * self.hy_inv**2
        
                y[i-stencil, j-stencil, 2] = pd[i, j]
                
                y[i-stencil, j-stencil, 3] = fp[i, j] \
                             + ( \
                                   - 1. * pd[i-1, j] \
                                   + 2. * pd[i,   j] \
                                   - 1. * pd[i+1, j] \
                               ) * self.hx_inv**2 \
                             + ( \
                                   - 1. * pd[i, j-1] \
                                   + 2. * pd[i, j  ] \
                                   - 1. * pd[i, j+1] \
                               ) * self.hy_inv**2
        
    
    def mult(self, Mat mat, Vec Q, Vec Y):
        self.matrix_mult(Q, Y)
        
    
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def matrix_mult(self, Vec Q, Vec Y):
        cdef int i, j, stencil
        cdef int xs, xe, ys, ye
        cdef double jpp, jpc, jcp
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        stencil = self.da1.getStencilWidth()
         
        self.da1.globalToLocal(Q, self.localQ)
         
        cdef double[:,:] q  = self.da1.getVecArray(self.localQ)[...]
        cdef double[:,:] t  = self.da1.getVecArray(self.T)[...]
        cdef double[:,:] y  = self.da1.getVecArray(Y)[...]
         
        cdef double[:,:] aa = self.aa
        
     
        for i in range(stencil, xe-xs+stencil):
            for j in range(stencil, ye-ys+stencil):
                jpp = (aa[i+1, j  ] - aa[i-1, j  ]) * (q[i,   j+1] - q[i,   j-1]) \
                    - (aa[i,   j+1] - aa[i,   j-1]) * (q[i+1, j  ] - q[i-1, j  ])
                
                jpc = aa[i+1, j  ] * (q[i+1, j+1] - q[i+1, j-1]) \
                    - aa[i-1, j  ] * (q[i-1, j+1] - q[i-1, j-1]) \
                    - aa[i,   j+1] * (q[i+1, j+1] - q[i-1, j+1]) \
                    + aa[i,   j-1] * (q[i+1, j-1] - q[i-1, j-1])
                
                jcp = aa[i+1, j+1] * (q[i,   j+1] - q[i+1, j  ]) \
                    - aa[i-1, j-1] * (q[i-1, j  ] - q[i,   j-1]) \
                    - aa[i-1, j+1] * (q[i,   j+1] - q[i-1, j  ]) \
                    + aa[i+1, j-1] * (q[i+1, j  ] - q[i,   j-1])
                
                t[i-stencil, j-stencil] = (jpp + jpc + jcp)
        
        
        self.da1.globalToLocal(self.T, self.localT)
        t  = self.da1.getVecArray(self.localT)[...]
        
        
        for i in range(stencil, xe-xs+stencil):
            for j in range(stencil, ye-ys+stencil):
                jpp = (aa[i+1, j  ] - aa[i-1, j  ]) * (t[i,   j+1] - t[i,   j-1]) \
                    - (aa[i,   j+1] - aa[i,   j-1]) * (t[i+1, j  ] - t[i-1, j  ])
                
                jpc = aa[i+1, j  ] * (t[i+1, j+1] - t[i+1, j-1]) \
                    - aa[i-1, j  ] * (t[i-1, j+1] - t[i-1, j-1]) \
                    - aa[i,   j+1] * (t[i+1, j+1] - t[i-1, j+1]) \
                    + aa[i,   j-1] * (t[i+1, j-1] - t[i-1, j-1])
                
                jcp = aa[i+1, j+1] * (t[i,   j+1] - t[i+1, j  ]) \
                    - aa[i-1, j-1] * (t[i-1, j  ] - t[i,   j-1]) \
                    - aa[i-1, j+1] * (t[i,   j+1] - t[i-1, j  ]) \
                    + aa[i+1, j-1] * (t[i+1, j  ] - t[i,   j-1])
                
                y[i-stencil, j-stencil] = q[i, j] - self.arakawa_fac2 * (jpp + jpc + jcp)
