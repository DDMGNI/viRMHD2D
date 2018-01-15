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
from rmhd.solvers.nonlinear.PETScNonlinearSolverArakawaJ1CFD2DB          import PETScSolverDB
from rmhd.solvers.preconditioner.PETScPreconditionerArakawaJ1CFD2DOF2Vec import PETScPreconditioner


class rmhd2d_ppc(rmhd2d):
    '''
    PETSc/Python Reduced MHD Solver in 2D using physics based preconditioner.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        super().__init__(cfgfile, mode = "ppc")
        
        
        OptDB = PETSc.Options()
         
#         OptDB.setValue('ksp_monitor',  '')
#         OptDB.setValue('snes_monitor', '')
#         
#         OptDB.setValue('log_info',    '')
#         OptDB.setValue('log_summary', '')
# 
#         OptDB.setValue('ksp_initial_guess_nonzero', 1)
        
#         OptDB.setValue('pc_type', 'hypre')
#         OptDB.setValue('pc_hypre_type', 'boomeramg')
        OptDB.setValue('pc_hypre_boomeramg_max_iter', 2)
#         OptDB.setValue('pc_hypre_boomeramg_max_levels', 6)
#         OptDB.setValue('pc_hypre_boomeramg_tol',  1e-7)

        
        # create Jacobian, Function, and linear Matrix objects
        if self.cfg["solver"]["preconditioner"] == 'none' or self.cfg["solver"]["preconditioner"] == None:
            self.petsc_precon   = None
        else:
            self.petsc_precon   = PETScPreconditioner(self.da1, self.da4, self.nx, self.ny, self.ht, self.hx, self.hy, self.de)

            self.petsc_precon.set_tolerances(poisson_rtol=self.cfg['solver']['pc_poisson_rtol'],
                                             poisson_atol=self.cfg['solver']['pc_poisson_atol'],
                                             poisson_max_it=self.cfg['solver']['pc_poisson_max_iter'],
                                             parabol_rtol=self.cfg['solver']['pc_parabol_rtol'],
                                             parabol_atol=self.cfg['solver']['pc_parabol_atol'],
                                             parabol_max_it=self.cfg['solver']['pc_parabol_max_iter'],
                                             jacobi_max_it=self.cfg['solver']['pc_jacobi_max_iter'])
            
        if self.nu != 0.:
            self.petsc_solver   = PETScSolverDB(self.da1, self.da4, self.nx, self.ny, self.ht, self.hx, self.hy, self.de, self.petsc_precon, self.nu)
        else:
            self.petsc_solver   = PETScSolver(self.da1, self.da4, self.nx, self.ny, self.ht, self.hx, self.hy, self.de, self.petsc_precon)
            
        
        # initialise matrixfree Jacobian
        self.Jmf = PETSc.Mat().createPython([self.x.getSizes(), self.b.getSizes()], 
                                            context=self.petsc_solver,
                                            comm=PETSc.COMM_WORLD)
        self.Jmf.setUp()
#         self.Jmf.setNullSpace(self.solver_nullspace)
        
        # create PC shell
#         self.pc = PETSc.PC().createPython(context=self.petsc_precon,
#                                           comm=PETSc.COMM_WORLD)
#         self.pc.setFromOptions()
#         self.pc.setUp()
        
        # create linear solver
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setOperators(self.Jmf)
        self.ksp.setInitialGuessNonzero(True)
        self.ksp.setType('fgmres')
        self.ksp.getPC().setType('none')
#         self.ksp.getPC().setType(PETSc.PC.Type.SHELL)
#         self.ksp.setPC(PETSc.PCShell(self.petsc_precon))
#         self.ksp.setPC(self.pc)
#         self.ksp.setPC(self.ksp.getPC().createPython(context=self.petsc_precon, comm=PETSc.COMM_WORLD))
#         self.ksp.setPCSide(PETSc.KSP.PCSide.RIGHT)
        self.ksp.setUp()

        
        # update solution history
        self.petsc_solver.update_previous(self.x)
        
        
    
    def __del__(self):
        self.ksp.destroy()
        self.Jmf.destroy()
    
    
    def run(self):
        
        run_time = time.time()
        
        alpha = 1.5
        gamma = 0.9
        ksp_rtol_max = 1E-3
        
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                print
            
            # calculate initial guess
            if self.cfg['solver']['petsc_snes_initial_guess']:
                self.calculate_initial_guess(initial=itime==1)
#                 self.calculate_initial_guess(initial=True)
            
            # update history
            self.petsc_solver.update_history()
            
            # copy initial guess to x
            if self.cfg['solver']['petsc_snes_initial_guess']:
                self.copy_x_from_da1_to_da4()
            
            # solve
            i = 0
            
            self.petsc_solver.update_previous(self.x)
            
            self.petsc_solver.function(self.f)
            pred_norm = self.f.norm()
            prev_norm = pred_norm
            
            tolerance = self.tolerance + self.cfg['solver']['petsc_snes_rtol'] * pred_norm 
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Nonlinear Solver Iteration %i:                           residual = %22.16E" % (i, pred_norm))
            
            while True:
            
                i+=1
                
                self.f.copy(self.b)
                self.b.scale(-1.)
                
                if self.petsc_precon == None:
                    self.dx.set(0.)
                else:
                    self.b.copy(self.dy)
                
                if self.cfg['solver']['petsc_ksp_adapt_rtol']:
                    if i == 1:
                        zeta_A  = 0.
                        zeta_B  = 0.
                        zeta_C  = 0.
                        zeta_D  = 0.
                        ksp_tol = self.cfg['solver']['petsc_ksp_rtol']
                    else:
                        zeta_A  = gamma * np.power(pred_norm / prev_norm , alpha)
                        zeta_B  = np.power(ksp_tol, alpha)
                        zeta_C  = np.min([ksp_rtol_max, np.max(zeta_A, zeta_B)])
                        zeta_D  = gamma * tolerance / pred_norm
                        ksp_tol = np.min([ksp_rtol_max, np.max(zeta_C, zeta_D)])
                    
                    self.ksp.setTolerances(rtol=ksp_tol)
                
                # solve linear system
                
                if self.petsc_precon == None:
                    self.ksp.solve(self.b, self.dx)
                else:
                    self.ksp.solve(self.b, self.dy)
                    self.petsc_precon.solve(self.dy, self.dx)
                
                self.x.axpy(1., self.dx)
                
                self.petsc_solver.update_previous(self.x)
                
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
            
