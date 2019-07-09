'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from run_rmhd2d import rmhd2d

import numpy as np
from numpy import abs

import time

from petsc4py import PETSc

from rmhd.solvers.common.PETScDerivatives                                import PETScDerivatives
from rmhd.solvers.linear.PETScPoissonCFD2                                import PETScPoisson
from rmhd.solvers.nonlinear.PETScNonlinearSolverArakawaJ1CFD2            import PETScSolver
from rmhd.solvers.nonlinear.PETScNonlinearSolverArakawaJ1CFD2DOF2        import PETScSolverDOF2
from rmhd.solvers.preconditioner.PETScPreconditionerArakawaJ1CFD2DOF2Vec import PETScPreconditioner


class rmhd2d_split(rmhd2d):
    '''
    PETSc/Python Reduced MHD Solver in 2D using split solver.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        super().__init__(cfgfile, mode = "split")
        
        
        OptDB = PETSc.Options()
        
#         OptDB.setValue('ksp_monitor',  '')
#         OptDB.setValue('snes_monitor', '')
#         
#         OptDB.setValue('log_info',    '')
#         OptDB.setValue('log_summary', '')

        OptDB.setValue('ksp_rtol',   self.cfg['solver']['petsc_ksp_rtol'])
        OptDB.setValue('ksp_atol',   self.cfg['solver']['petsc_ksp_atol'])
        OptDB.setValue('ksp_max_it', self.cfg['solver']['petsc_ksp_max_iter'])
#         OptDB.setValue('ksp_initial_guess_nonzero', 1)
        
        OptDB.setValue('pc_type', 'hypre')
        OptDB.setValue('pc_hypre_type', 'boomeramg')
        OptDB.setValue('pc_hypre_boomeramg_max_iter', 2)
#         OptDB.setValue('pc_hypre_boomeramg_max_levels', 6)
#         OptDB.setValue('pc_hypre_boomeramg_tol',  1e-7)
        
        
        # create DA (dof = 2 for A, P)
        self.da2 = PETSc.DA().create(dim=2, dof=2,
                                     sizes=[self.nx, self.ny],
                                     proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                     boundary_type=('periodic', 'periodic'),
                                     stencil_width=1,
                                     stencil_type='box')
        
        # create solution and RHS vector
        self.dx2 = self.da2.createGlobalVec()
        self.dy2 = self.da2.createGlobalVec()
        self.b   = self.da2.createGlobalVec()
        
        self.Ad = self.da1.createGlobalVec()
        self.Jd = self.da1.createGlobalVec()
        self.Pd = self.da1.createGlobalVec()
        self.Od = self.da1.createGlobalVec()
        
        # create Jacobian, Function, and linear Matrix objects
        self.petsc_precon   = PETScPreconditioner(self.da1, self.da2, self.nx, self.ny, self.ht, self.hx, self.hy)
#         self.petsc_solver2  = PETScSolverDOF2(self.da1, self.da2, self.nx, self.ny, self.ht, self.hx, self.hy)
        self.petsc_solver2  = PETScSolverDOF2(self.da1, self.da2, self.nx, self.ny, self.ht, self.hx, self.hy, self.petsc_precon)
        self.petsc_solver  = PETScSolver(self.da1, self.da4, self.nx, self.ny, self.ht, self.hx, self.hy)
        
        
        self.petsc_precon.set_tolerances(poisson_rtol=self.cfg['solver']['pc_poisson_rtol'],
                                         poisson_atol=self.cfg['solver']['pc_poisson_atol'],
                                         poisson_max_it=self.cfg['solver']['pc_poisson_max_iter'],
                                         parabol_rtol=self.cfg['solver']['pc_parabol_rtol'],
                                         parabol_atol=self.cfg['solver']['pc_parabol_atol'],
                                         parabol_max_it=self.cfg['solver']['pc_parabol_max_iter'],
                                         jacobi_max_it=self.cfg['solver']['pc_jacobi_max_iter'])
        
        # initialise matrixfree Jacobian
        self.Jmf = PETSc.Mat().createPython([self.b.getSizes(), self.b.getSizes()], 
                                            context=self.petsc_solver2,
                                            comm=PETSc.COMM_WORLD)
        self.Jmf.setUp()
        
        # create linear solver
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setOperators(self.Jmf)
        self.ksp.setInitialGuessNonzero(True)
        self.ksp.setType('fgmres')
        self.ksp.getPC().setType('none')
        
        # update solution history
        self.petsc_solver.update_previous(self.x)
        self.petsc_solver2.update_previous(self.A, self.J, self.P, self.O)
        
        
    
    def __del__(self):
        self.ksp.destroy()
        self.Jmf.destroy()
    
    
    def run(self):
        
        run_time = time.time()
        
        alpha = 1.5 # 64x64
#         alpha = 1.1  # 128x128
#         alpha = 1.5  # 256x256
        gamma = 0.9
#         ksp_max = 1E-1  # 64x64, 128x128
        ksp_max = 1E-3 # 256x256
        
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                print
            
            # calculate initial guess
            self.calculate_initial_guess(initial=itime==1)
#             self.calculate_initial_guess(initial=True)
            
            # update history
            self.petsc_solver.update_history()
            self.petsc_solver2.update_history()
            
            # copy initial guess to x
            x_arr = self.da4.getVecArray(self.x)
            x_arr[:,:,0] = self.da1.getVecArray(self.A)[:,:]
            x_arr[:,:,1] = self.da1.getVecArray(self.J)[:,:]
            x_arr[:,:,2] = self.da1.getVecArray(self.P)[:,:]
            x_arr[:,:,3] = self.da1.getVecArray(self.O)[:,:]
            
            # solve
            i = 0
            
            self.petsc_solver.update_previous(self.x)
            self.petsc_solver2.update_previous(self.A, self.J, self.P, self.O)
            
            self.petsc_solver.function(self.f)
            pred_norm = self.f.norm()
            prev_norm = pred_norm
            
            tolerance = self.tolerance + self.cfg['solver']['petsc_snes_rtol'] * pred_norm 
#             print("tolerance:", self.tolerance, self.cfg['solver']['petsc_snes_rtol'] * pred_norm, tolerance)
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Nonlinear Solver Iteration %i:                           residual = %22.16E" % (i, pred_norm))
            
            while True:
            
                i+=1
                
                f_arr = self.da4.getVecArray(self.f)
                b_arr = self.da2.getVecArray(self.b)
                
                b_arr[:,:,0] = -f_arr[:,:,0]
                b_arr[:,:,1] = -f_arr[:,:,3]
                
                self.da1.getVecArray(self.FA)[...] = f_arr[:,:,0]
                self.da1.getVecArray(self.FJ)[...] = f_arr[:,:,1]
                self.da1.getVecArray(self.FP)[...] = f_arr[:,:,2]
                self.da1.getVecArray(self.FO)[...] = f_arr[:,:,3]
                
                self.petsc_solver2.update_function(self.FA, self.FJ, self.FP, self.FO)
                
                self.dy2.set(0.)
#                 self.b.copy(self.dy2)

                if i == 1:
                    zeta_A  = 0.
                    zeta_B  = 0.
                    zeta_C  = 0.
                    zeta_D  = 0.
                    ksp_tol = self.cfg['solver']['petsc_ksp_rtol']
#                     self.ksp.setTolerances(rtol=ksp_tol, max_it=3)
                else:
                    zeta_A  = gamma * np.power(pred_norm / prev_norm , alpha)
                    zeta_B  = np.power(ksp_tol, alpha)
                    zeta_C  = min(ksp_max, max(zeta_A, zeta_B))
                    zeta_D  = gamma * tolerance / pred_norm
                    ksp_tol = min(ksp_max, max(zeta_C, zeta_D))
#                     self.ksp.setTolerances(rtol=ksp_tol, max_it=5)
                
#                 self.ksp.setTolerances(rtol=ksp_tol)
                self.ksp.solve(self.b, self.dy2)
                
                self.petsc_precon.solve(self.dy2, self.dx2)
#                 self.dy2.copy(self.dx2)

                dx_arr = self.da2.getVecArray(self.dx2)

                self.da1.getVecArray(self.Ad)[...] = dx_arr[:,:,0]
                self.da1.getVecArray(self.Pd)[...] = dx_arr[:,:,1]
                
                self.derivatives.laplace_vec(self.Pd, self.Od, -1.)
                self.derivatives.laplace_vec(self.Ad, self.Jd, -1.)
                
                self.Od.axpy(-1., self.FP)
                self.Jd.axpy(-1., self.FJ)

                dx_arr = self.da4.getVecArray(self.dx)
                dx_arr[:,:,0] = self.da1.getVecArray(self.Ad)[...]
                dx_arr[:,:,1] = self.da1.getVecArray(self.Jd)[...]
                dx_arr[:,:,2] = self.da1.getVecArray(self.Pd)[...]
                dx_arr[:,:,3] = self.da1.getVecArray(self.Od)[...]
                
                self.x.axpy(1., self.dx)
                self.A.axpy(1., self.Ad)
                self.J.axpy(1., self.Jd)
                self.P.axpy(1., self.Pd)
                self.O.axpy(1., self.Od)
                
                self.petsc_solver.update_previous(self.x)
                self.petsc_solver2.update_previous(self.A, self.J, self.P, self.O)
                
                prev_norm = pred_norm
                self.petsc_solver.function(self.f)
                pred_norm = self.f.norm()

                if PETSc.COMM_WORLD.getRank() == 0:
                    print("  Nonlinear Solver Iteration %i: %5i GMRES iterations,   residual = %22.16E,   tolerance = %22.16E" % (i, self.ksp.getIterationNumber(), pred_norm, self.ksp.getTolerances()[0]) )
                
                if abs(prev_norm - pred_norm) < self.cfg['solver']['petsc_snes_stol'] or pred_norm < tolerance or i >= self.cfg['solver']['petsc_snes_max_iter']:
                    break
            
            # output some solver info
            if PETSc.COMM_WORLD.getRank() == 0:
                print()
            
            
            # save to hdf5 file
            self.save_to_hdf5(itime)
        
        # output total time spent in run
        run_time = time.time() - run_time

        if PETSc.COMM_WORLD.getRank() == 0:
            print("Solver runtime: %f seconds." % run_time)
            print()
            
