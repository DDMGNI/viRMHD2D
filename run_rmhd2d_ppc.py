'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from run_rmhd2d import rmhd2d

import numpy as np
from numpy import abs

import argparse, sys, time
import pstats, cProfile

from config import Config

from petsc4py import PETSc

from PETScDerivatives                    import PETScDerivatives
from PETScPoissonCFD2                    import PETScPoisson
from PETScNonlinearSolverArakawaJ1CFD2   import PETScSolver
from PETScPreconditionerArakawaJ1CFD2    import PETScPreconditioner
# from PETScPreconditionerArakawaJ1CFD2Vec import PETScPreconditioner


solver_package = 'superlu_dist'
# solver_package = 'mumps'
# solver_package = 'pastix'


class rmhd2d_ppc(rmhd2d):
    '''
    PETSc/Python Reduced MHD Solver in 2D.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        super().__init__(cfgfile)#rmhd2d_ppc, self
        
        OptDB = PETSc.Options()
        
#         OptDB.setValue('ksp_monitor',  '')
#         OptDB.setValue('snes_monitor', '')
#         
#         OptDB.setValue('log_info',    '')
#         OptDB.setValue('log_summary', '')

#         OptDB.setValue('snes_ls', 'basic')
#         OptDB.setValue('snes_ls', 'quadratic')
# 
#         OptDB.setValue('pc_asm_type',  'restrict')
#         OptDB.setValue('pc_asm_overlap', 3)
#         OptDB.setValue('sub_ksp_type', 'preonly')
#         OptDB.setValue('sub_pc_type', 'lu')
#         OptDB.setValue('sub_pc_factor_mat_solver_package', 'mumps')
        
#         OptDB.setValue('snes_rtol',   self.cfg['solver']['petsc_snes_rtol'])
#         OptDB.setValue('snes_atol',   self.cfg['solver']['petsc_snes_atol'])
#         OptDB.setValue('snes_stol',   self.cfg['solver']['petsc_snes_stol'])
#         OptDB.setValue('snes_max_it', self.cfg['solver']['petsc_snes_max_iter'])
        
        OptDB.setValue('ksp_rtol',   self.cfg['solver']['petsc_ksp_rtol'])
        OptDB.setValue('ksp_atol',   self.cfg['solver']['petsc_ksp_atol'])
        OptDB.setValue('ksp_max_it', self.cfg['solver']['petsc_ksp_max_iter'])
#         OptDB.setValue('ksp_initial_guess_nonzero', 1)
        
        
#         OptDB.setValue('ksp_type', 'fgmres')
#         OptDB.setValue('pc_type', 'gamg')
#         OptDB.setValue('pc_type', 'ml')
        OptDB.setValue('pc_type', 'hypre')
        OptDB.setValue('pc_hypre_type', 'boomeramg')
#         OptDB.setValue('pc_hypre_boomeramg_tol',  1e-7)
        OptDB.setValue('pc_hypre_boomeramg_max_iter', 2)
#         OptDB.setValue('pc_hypre_boomeramg_max_levels', 6)
        
#         OptDB.setValue('pc_hypre_type', 'parasails')
#         OptDB.setValue('da_refine', 1)
#         OptDB.setValue('pc_mg_levels', 3)
# #         OptDB.setValue('pc_mg_type', 'full')
#         OptDB.setValue('mg_coarse_ksp_type', 'cg')
#         OptDB.setValue('mg_coarse_pc_type', 'jacobi')
#         OptDB.setValue('mg_coarse_ksp_max_it', 10)
# #         OptDB.setValue('mg_coarse_ksp_type', 'preonly')
# #         OptDB.setValue('mg_coarse_pc_type', 'lu')
# #         OptDB.setValue('mg_coarse_pc_factor_shift_type', 'nonzero')
# #         OptDB.setValue('mg_levels_ksp_type', 'richardson')
#         OptDB.setValue('mg_levels_ksp_type', 'chebyshev')
#         OptDB.setValue('mg_levels_pc_type', 'jacobi')
# #         OptDB.setValue('mg_levels_pc_type', 'sor')
#         OptDB.setValue('mg_levels_ksp_max_it', 10)
        
# #        OptDB.setValue('mat_superlu_dist_matinput', 'DISTRIBUTED')
# #        OptDB.setValue('mat_superlu_dist_rowperm',  'NATURAL')
#         OptDB.setValue('mat_superlu_dist_colperm',  'PARMETIS')
#         OptDB.setValue('mat_superlu_dist_parsymbfact', 1)
        
        
        
        
        # create Jacobian, Function, and linear Matrix objects
        self.petsc_precon   = PETScPreconditioner(self.da1, self.da4, self.nx, self.ny, self.ht, self.hx, self.hy)
        self.petsc_solver   = PETScSolver(self.da1, self.da4, self.nx, self.ny, self.ht, self.hx, self.hy, self.petsc_precon)
        
        
        self.petsc_precon.set_tolerances(poisson_rtol=self.cfg['solver']['pc_poisson_rtol'],
                                         poisson_atol=self.cfg['solver']['pc_poisson_atol'],
                                         poisson_max_it=self.cfg['solver']['pc_poisson_max_iter'],
                                         parabol_rtol=self.cfg['solver']['pc_parabol_rtol'],
                                         parabol_atol=self.cfg['solver']['pc_parabol_atol'],
                                         parabol_max_it=self.cfg['solver']['pc_parabol_max_iter'],
                                         jacobi_max_it=self.cfg['solver']['pc_jacobi_max_iter'])
        
        # initialise matrixfree Jacobian
        self.Jmf = PETSc.Mat().createPython([self.x.getSizes(), self.b.getSizes()], 
                                            context=self.petsc_solver,
                                            comm=PETSc.COMM_WORLD)
        self.Jmf.setUp()
        
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

        
        # update solution history
        self.petsc_solver.update_previous(self.x)
        
        
        
    
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
            current_time = self.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
                print
                self.time.setValue(0, current_time)
            
            # calculate initial guess
            self.calculate_initial_guess(initial=itime==1)
#             self.calculate_initial_guess(initial=True)
            
            # update history
            self.petsc_solver.update_history()
            
            # copy initial guess to x
            x_arr = self.da4.getVecArray(self.x)
            x_arr[:,:,0] = self.da1.getVecArray(self.A)[:,:]
            x_arr[:,:,1] = self.da1.getVecArray(self.J)[:,:]
            x_arr[:,:,2] = self.da1.getVecArray(self.P)[:,:]
            x_arr[:,:,3] = self.da1.getVecArray(self.O)[:,:]
            
            # solve
            i = 0
            
            self.petsc_solver.update_previous(self.x)
            
            self.petsc_solver.function(self.f)
            pred_norm = self.f.norm()
            prev_norm = pred_norm
            
            tolerance = self.tolerance + self.cfg['solver']['petsc_snes_rtol'] * pred_norm 
#             print("tolerance:", self.tolerance, self.cfg['solver']['petsc_snes_rtol'] * pred_norm, tolerance)
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Nonlinear Solver Iteration %i:                           residual = %22.16E" % (i, pred_norm))
            
            while True:
            
                i+=1
                
                self.f.copy(self.b)
                self.b.scale(-1.)
#                 self.dy.set(0.)
                self.b.copy(self.dy)

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
                    zeta_C  = np.min([ksp_max, np.max(zeta_A, zeta_B)])
                    zeta_D  = gamma * tolerance / pred_norm
                    ksp_tol = np.min([ksp_max, np.max(zeta_C, zeta_D)])
#                     self.ksp.setTolerances(rtol=ksp_tol, max_it=5)
                
                self.ksp.setTolerances(rtol=ksp_tol)
                self.ksp.solve(self.b, self.dy)
#                 self.ksp.solve(self.b, self.dx)
                
#                 if PETSc.COMM_WORLD.getRank() == 0:
#                     print(" PC solve")
                
                self.petsc_precon.solve(self.dy, self.dx)
                
                self.x.axpy(1., self.dx)
                
                self.petsc_solver.update_previous(self.x)
                
                prev_norm = pred_norm
                self.petsc_solver.function(self.f)
                pred_norm = self.f.norm()

                if PETSc.COMM_WORLD.getRank() == 0:
                    print("  Nonlinear Solver Iteration %i: %5i GMRES iterations,   residual = %22.16E,   tolerance = %22.16E" % (i, self.ksp.getIterationNumber(), pred_norm, ksp_tol) )
                
                if abs(prev_norm - pred_norm) < self.cfg['solver']['petsc_snes_stol'] or pred_norm < tolerance or i >= self.cfg['solver']['petsc_snes_max_iter']:
                    break
            
            # output some solver info
            if PETSc.COMM_WORLD.getRank() == 0:
                print()
            
            
            # save to hdf5 file
            if itime % self.nsave == 0 or itime == self.nt + 1:
                self.save_to_hdf5(itime)
        
        # output total time spent in run
        run_time = time.time() - run_time

        if PETSc.COMM_WORLD.getRank() == 0:
            print("Solver runtime: %f seconds." % run_time)
            print()
            
        
    
    

    

if __name__ == '__main__':

    OptDB = PETSc.Options()
 
#     parser = argparse.ArgumentParser(description='PETSc MHD Solver in 2D')
#     parser.add_argument('-prof','--profiler', action='store_true', required=False,
#                         help='Activate Profiler')
#     parser.add_argument('-jac','--jacobian', action='store_true', required=False,
#                         help='Check Jacobian')
#     parser.add_argument('runfile', metavar='runconfig', type=str,
#                         help='Run Configuration File')
#     
#     args = parser.parse_args()
    
    runfile = OptDB.getString('c')
    petscvp = petscMHD2D(runfile)

#     if args.profiler:
    if OptDB.getBool('profiler', default=False):
        cProfile.runctx("petscvp.run()", globals(), locals(), "profile.prof")
        
        if PETSc.COMM_WORLD.getRank() == 0:
            s = pstats.Stats("profile.prof")
            s.strip_dirs().sort_stats("time").print_stats()
    elif OptDB.getBool('jacobian', default=False):
        petscvp.check_jacobian()
    else:
        petscvp.run()
