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
        self.parabol_ksp.setOperators(self.Qm)
        self.parabol_ksp.setType('cg')
        self.parabol_ksp.getPC().setType('none')
#         self.parabol_ksp.setNormType(self.parabol_ksp.NormType.NORM_NONE)
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
    def solve(self, Vec F, Vec Y):
        
        self.update_function(F)

        cdef int i, j, stencil
        cdef int ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        cdef double jpp, jpc, jcp
        cdef double arakawa_fac = 0.5 * self.ht * self.hx_inv * self.hy_inv / 12.
 
        (xs, xe), (ys, ye) = self.da1.getRanges()
        stencil = self.da1.getStencilWidth()
        
        
        self.da1.globalToLocal(self.Aa, self.localAa)
        self.da1.globalToLocal(self.Pa, self.localPa)
        
        self.da1.globalToLocal(self.FA, self.localFA)
        self.da1.globalToLocal(self.FJ, self.localFJ)
        self.da1.globalToLocal(self.FP, self.localFP)
        self.da1.globalToLocal(self.FO, self.localFO)
         
        cdef double[:,:] aa = self.da1.getVecArray(self.localAa)[...]
        cdef double[:,:] pa = self.da1.getVecArray(self.localPa)[...]
        
        cdef double[:,:] fa = self.da1.getVecArray(self.localFA)[...]
        cdef double[:,:] fj = self.da1.getVecArray(self.localFJ)[...]
        cdef double[:,:] fp = self.da1.getVecArray(self.localFP)[...]
        cdef double[:,:] fo = self.da1.getVecArray(self.localFO)[...]
        
        cdef double[:,:] pb = self.da1.getVecArray(self.Pb)[...]
        cdef double[:,:] qb

        cdef double[:,:] l
        cdef double[:,:] t
        cdef double[:,:] ad
        cdef double[:,:] pd
        
        
        for i in range(xs, xe):
            ix = i-xs+stencil
            iy = i-xs
              
            for j in range(ys, ye):
                jx = j-ys+stencil
                jy = j-ys
                
                pb[iy, jy] = fo[ix, jx] - fp[ix, jx]
                
                jpp = (pa[ix+1, jx  ] - pa[ix-1, jx  ]) * (fp[ix,   jx+1] - fp[ix,   jx-1]) \
                    - (pa[ix,   jx+1] - pa[ix,   jx-1]) * (fp[ix+1, jx  ] - fp[ix-1, jx  ])
                 
                jpc = pa[ix+1, jx  ] * (fp[ix+1, jx+1] - fp[ix+1, jx-1]) \
                    - pa[ix-1, jx  ] * (fp[ix-1, jx+1] - fp[ix-1, jx-1]) \
                    - pa[ix,   jx+1] * (fp[ix+1, jx+1] - fp[ix-1, jx+1]) \
                    + pa[ix,   jx-1] * (fp[ix+1, jx-1] - fp[ix-1, jx-1])
                 
                jcp = pa[ix+1, jx+1] * (fp[ix,   jx+1] - fp[ix+1, jx  ]) \
                    - pa[ix-1, jx-1] * (fp[ix-1, jx  ] - fp[ix,   jx-1]) \
                    - pa[ix-1, jx+1] * (fp[ix,   jx+1] - fp[ix-1, jx  ]) \
                    + pa[ix+1, jx-1] * (fp[ix+1, jx  ] - fp[ix,   jx-1])
                 
                pb[iy, jy] += arakawa_fac * (jpp + jpc + jcp)
         
                 
                jpp = (aa[ix+1, jx  ] - aa[ix-1, jx  ]) * (fj[ix,   jx+1] - fj[ix,   jx-1]) \
                    - (aa[ix,   jx+1] - aa[ix,   jx-1]) * (fj[ix+1, jx  ] - fj[ix-1, jx  ])
                 
                jpc = aa[ix+1, jx  ] * (fj[ix+1, jx+1] - fj[ix+1, jx-1]) \
                    - aa[ix-1, jx  ] * (fj[ix-1, jx+1] - fj[ix-1, jx-1]) \
                    - aa[ix,   jx+1] * (fj[ix+1, jx+1] - fj[ix-1, jx+1]) \
                    + aa[ix,   jx-1] * (fj[ix+1, jx-1] - fj[ix-1, jx-1])
                 
                jcp = aa[ix+1, jx+1] * (fj[ix,   jx+1] - fj[ix+1, jx  ]) \
                    - aa[ix-1, jx-1] * (fj[ix-1, jx  ] - fj[ix,   jx-1]) \
                    - aa[ix-1, jx+1] * (fj[ix,   jx+1] - fj[ix-1, jx  ]) \
                    + aa[ix+1, jx-1] * (fj[ix+1, jx  ] - fj[ix,   jx-1])
                 
                pb[iy, jy] -= arakawa_fac * (jpp + jpc + jcp)
                        
        
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
                
                for i in range(xs, xe):
                    ix = i-xs+stencil
                    iy = i-xs
                      
                    for j in range(ys, ye):
                        jx = j-ys+stencil
                        jy = j-ys
                        
                        jpp = (pa[ix+1, jx  ] - pa[ix-1, jx  ]) * (pd[ix,   jx+1] - pd[ix,   jx-1]) \
                            - (pa[ix,   jx+1] - pa[ix,   jx-1]) * (pd[ix+1, jx  ] - pd[ix-1, jx  ])
                         
                        jpc = pa[ix+1, jx  ] * (pd[ix+1, jx+1] - pd[ix+1, jx-1]) \
                            - pa[ix-1, jx  ] * (pd[ix-1, jx+1] - pd[ix-1, jx-1]) \
                            - pa[ix,   jx+1] * (pd[ix+1, jx+1] - pd[ix-1, jx+1]) \
                            + pa[ix,   jx-1] * (pd[ix+1, jx-1] - pd[ix-1, jx-1])
                         
                        jcp = pa[ix+1, jx+1] * (pd[ix,   jx+1] - pd[ix+1, jx  ]) \
                            - pa[ix-1, jx-1] * (pd[ix-1, jx  ] - pd[ix,   jx-1]) \
                            - pa[ix-1, jx+1] * (pd[ix,   jx+1] - pd[ix-1, jx  ]) \
                            + pa[ix+1, jx-1] * (pd[ix+1, jx  ] - pd[ix,   jx-1])
                         
                        t[iy, jy] = (jpp + jpc + jcp)
            
            
            self.da1.globalToLocal(self.T, self.localT)
            t  = self.da1.getVecArray(self.localT)[...]
            qb = self.da1.getVecArray(self.Qb)[...]
         
            for i in range(xs, xe):
                ix = i-xs+stencil
                iy = i-xs
                  
                for j in range(ys, ye):
                    jx = j-ys+stencil
                    jy = j-ys
                    
                    qb[iy, jy] = fa[ix, jx]
                    
                    jpp = (aa[ix+1, jx  ] - aa[ix-1, jx  ]) * (l[ix,   jx+1] - l[ix,   jx-1]) \
                        - (aa[ix,   jx+1] - aa[ix,   jx-1]) * (l[ix+1, jx  ] - l[ix-1, jx  ])
                     
                    jpc = aa[ix+1, jx  ] * (l[ix+1, jx+1] - l[ix+1, jx-1]) \
                        - aa[ix-1, jx  ] * (l[ix-1, jx+1] - l[ix-1, jx-1]) \
                        - aa[ix,   jx+1] * (l[ix+1, jx+1] - l[ix-1, jx+1]) \
                        + aa[ix,   jx-1] * (l[ix+1, jx-1] - l[ix-1, jx-1])
                     
                    jcp = aa[ix+1, jx+1] * (l[ix,   jx+1] - l[ix+1, jx  ]) \
                        - aa[ix-1, jx-1] * (l[ix-1, jx  ] - l[ix,   jx-1]) \
                        - aa[ix-1, jx+1] * (l[ix,   jx+1] - l[ix-1, jx  ]) \
                        + aa[ix+1, jx-1] * (l[ix+1, jx  ] - l[ix,   jx-1])
                     
                    qb[iy, jy] += arakawa_fac * (jpp + jpc + jcp)
                    
                    
                    jpp = (pa[ix+1, jx  ] - pa[ix-1, jx  ]) * (ad[ix,   jx+1] - ad[ix,   jx-1]) \
                        - (pa[ix,   jx+1] - pa[ix,   jx-1]) * (ad[ix+1, jx  ] - ad[ix-1, jx  ])
                     
                    jpc = pa[ix+1, jx  ] * (ad[ix+1, jx+1] - ad[ix+1, jx-1]) \
                        - pa[ix-1, jx  ] * (ad[ix-1, jx+1] - ad[ix-1, jx-1]) \
                        - pa[ix,   jx+1] * (ad[ix+1, jx+1] - ad[ix-1, jx+1]) \
                        + pa[ix,   jx-1] * (ad[ix+1, jx-1] - ad[ix-1, jx-1])
                     
                    jcp = pa[ix+1, jx+1] * (ad[ix,   jx+1] - ad[ix+1, jx  ]) \
                        - pa[ix-1, jx-1] * (ad[ix-1, jx  ] - ad[ix,   jx-1]) \
                        - pa[ix-1, jx+1] * (ad[ix,   jx+1] - ad[ix-1, jx  ]) \
                        + pa[ix+1, jx-1] * (ad[ix+1, jx  ] - ad[ix,   jx-1])
                     
                    qb[iy, jy] -= arakawa_fac * (jpp + jpc + jcp)
                    
             
                    jpp = (aa[ix+1, jx  ] - aa[ix-1, jx  ]) * (t[ix,   jx+1] - t[ix,   jx-1]) \
                        - (aa[ix,   jx+1] - aa[ix,   jx-1]) * (t[ix+1, jx  ] - t[ix-1, jx  ])
                     
                    jpc = aa[ix+1, jx  ] * (t[ix+1, jx+1] - t[ix+1, jx-1]) \
                        - aa[ix-1, jx  ] * (t[ix-1, jx+1] - t[ix-1, jx-1]) \
                        - aa[ix,   jx+1] * (t[ix+1, jx+1] - t[ix-1, jx+1]) \
                        + aa[ix,   jx-1] * (t[ix+1, jx-1] - t[ix-1, jx-1])
                     
                    jcp = aa[ix+1, jx+1] * (t[ix,   jx+1] - t[ix+1, jx  ]) \
                        - aa[ix-1, jx-1] * (t[ix-1, jx  ] - t[ix,   jx-1]) \
                        - aa[ix-1, jx+1] * (t[ix,   jx+1] - t[ix-1, jx  ]) \
                        + aa[ix+1, jx-1] * (t[ix+1, jx  ] - t[ix,   jx-1])
                     
                    qb[iy, jy] -= arakawa_fac * arakawa_fac * (jpp + jpc + jcp)
                    
                    
            self.parabol_ksp.solve(self.Qb, self.Ad)
            
            
            self.da1.globalToLocal(self.Ad, self.localAd)
            ad = self.da1.getVecArray(self.localAd)[...]
            pd = self.da1.getVecArray(self.Pd)[...]
            
            for i in range(xs, xe):
                ix = i-xs+stencil
                iy = i-xs
                
                for j in range(ys, ye):
                    jx = j-ys+stencil
                    jy = j-ys
                    
                    pd[iy, jy] = l[ix, jx] - arakawa_fac * t[ix, jx]
                    
                    jpp = (aa[ix+1, jx  ] - aa[ix-1, jx  ]) * (ad[ix,   jx+1] - ad[ix,   jx-1]) \
                        - (aa[ix,   jx+1] - aa[ix,   jx-1]) * (ad[ix+1, jx  ] - ad[ix-1, jx  ])
                     
                    jpc = aa[ix+1, jx  ] * (ad[ix+1, jx+1] - ad[ix+1, jx-1]) \
                        - aa[ix-1, jx  ] * (ad[ix-1, jx+1] - ad[ix-1, jx-1]) \
                        - aa[ix,   jx+1] * (ad[ix+1, jx+1] - ad[ix-1, jx+1]) \
                        + aa[ix,   jx-1] * (ad[ix+1, jx-1] - ad[ix-1, jx-1])
                     
                    jcp = aa[ix+1, jx+1] * (ad[ix,   jx+1] - ad[ix+1, jx  ]) \
                        - aa[ix-1, jx-1] * (ad[ix-1, jx  ] - ad[ix,   jx-1]) \
                        - aa[ix-1, jx+1] * (ad[ix,   jx+1] - ad[ix-1, jx  ]) \
                        + aa[ix+1, jx-1] * (ad[ix+1, jx  ] - ad[ix,   jx-1])
                     
                    pd[iy, jy] += arakawa_fac * (jpp + jpc + jcp)
                    
            self.da1.globalToLocal(self.Pd, self.localPd)
            pd = self.da1.getVecArray(self.localPd)[...]
            
        
        cdef double[:,:,:] y = self.da4.getVecArray(Y)[...]
        
        for i in range(xs, xe):
            ix = i-xs+stencil
            iy = i-xs
              
            for j in range(ys, ye):
                jx = j-ys+stencil
                jy = j-ys
                
                y[iy, jy, 0] = ad[ix, jx]
                
                y[iy, jy, 1] = fj[ix, jx] \
                             + ( \
                                   - 1. * ad[ix-1, jx] \
                                   + 2. * ad[ix,   jx] \
                                   - 1. * ad[ix+1, jx] \
                               ) * self.hx_inv**2 \
                             + ( \
                                   - 1. * ad[ix, jx-1] \
                                   + 2. * ad[ix, jx  ] \
                                   - 1. * ad[ix, jx+1] \
                               ) * self.hy_inv**2
        
                y[iy, jy, 2] = pd[ix, jx]
                
                y[iy, jy, 3] = fp[ix, jx] \
                             + ( \
                                   - 1. * pd[ix-1, jx] \
                                   + 2. * pd[ix,   jx] \
                                   - 1. * pd[ix+1, jx] \
                               ) * self.hx_inv**2 \
                             + ( \
                                   - 1. * pd[ix, jx-1] \
                                   + 2. * pd[ix, jx  ] \
                                   - 1. * pd[ix, jx+1] \
                               ) * self.hy_inv**2
        
    
    def mult(self, Mat mat, Vec Q, Vec Y):
        self.matrix_mult(Q, Y)
        
    
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def matrix_mult(self, Vec Q, Vec Y):
        cdef int i, j, stencil
        cdef int ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        cdef double jpp, jpc, jcp
        cdef double arakawa_fac = 0.5 * self.ht * self.hx_inv * self.hy_inv / 12.
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        stencil = self.da1.getStencilWidth()
         
        self.da1.globalToLocal(Q,       self.localQ)
        self.da1.globalToLocal(self.Aa, self.localAa)
         
        cdef double[:,:] q  = self.da1.getVecArray(self.localQ )[...]
        cdef double[:,:] aa = self.da1.getVecArray(self.localAa)[...]
        cdef double[:,:] t  = self.da1.getVecArray(self.T)[...]
        cdef double[:,:] y  = self.da1.getVecArray(Y)[...]
         
     
        for i in range(xs, xe):
            ix = i-xs+stencil
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+stencil
                jy = j-ys
                
                jpp = (aa[ix+1, jx  ] - aa[ix-1, jx  ]) * (q[ix,   jx+1] - q[ix,   jx-1]) \
                    - (aa[ix,   jx+1] - aa[ix,   jx-1]) * (q[ix+1, jx  ] - q[ix-1, jx  ])
                
                jpc = aa[ix+1, jx  ] * (q[ix+1, jx+1] - q[ix+1, jx-1]) \
                    - aa[ix-1, jx  ] * (q[ix-1, jx+1] - q[ix-1, jx-1]) \
                    - aa[ix,   jx+1] * (q[ix+1, jx+1] - q[ix-1, jx+1]) \
                    + aa[ix,   jx-1] * (q[ix+1, jx-1] - q[ix-1, jx-1])
                
                jcp = aa[ix+1, jx+1] * (q[ix,   jx+1] - q[ix+1, jx  ]) \
                    - aa[ix-1, jx-1] * (q[ix-1, jx  ] - q[ix,   jx-1]) \
                    - aa[ix-1, jx+1] * (q[ix,   jx+1] - q[ix-1, jx  ]) \
                    + aa[ix+1, jx-1] * (q[ix+1, jx  ] - q[ix,   jx-1])
                
                t[iy, jy] = arakawa_fac * (jpp + jpc + jcp)
        
        
        self.da1.globalToLocal(self.T, self.localT)
        t  = self.da1.getVecArray(self.localT)[...]
        
        
        for i in range(xs, xe):
            ix = i-xs+stencil
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+stencil
                jy = j-ys
                
                jpp = (aa[ix+1, jx  ] - aa[ix-1, jx  ]) * (t[ix,   jx+1] - t[ix,   jx-1]) \
                    - (aa[ix,   jx+1] - aa[ix,   jx-1]) * (t[ix+1, jx  ] - t[ix-1, jx  ])
                
                jpc = aa[ix+1, jx  ] * (t[ix+1, jx+1] - t[ix+1, jx-1]) \
                    - aa[ix-1, jx  ] * (t[ix-1, jx+1] - t[ix-1, jx-1]) \
                    - aa[ix,   jx+1] * (t[ix+1, jx+1] - t[ix-1, jx+1]) \
                    + aa[ix,   jx-1] * (t[ix+1, jx-1] - t[ix-1, jx-1])
                
                jcp = aa[ix+1, jx+1] * (t[ix,   jx+1] - t[ix+1, jx  ]) \
                    - aa[ix-1, jx-1] * (t[ix-1, jx  ] - t[ix,   jx-1]) \
                    - aa[ix-1, jx+1] * (t[ix,   jx+1] - t[ix-1, jx  ]) \
                    + aa[ix+1, jx-1] * (t[ix+1, jx  ] - t[ix,   jx-1])
                
                y[iy, jy] = q[ix, jx] - arakawa_fac * (jpp + jpc + jcp)
