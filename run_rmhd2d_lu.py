'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from run_rmhd2d import rmhd2d

from petsc4py import PETSc

import numpy as np

import argparse, time
import pstats, cProfile

from config import Config

from PETScDerivatives                  import PETScDerivatives
from PETScPoissonCFD2                  import PETScPoisson
from PETScNonlinearSolverArakawaJ1CFD2 import PETScSolver


solver_package = 'superlu_dist'
# solver_package = 'mumps'
# solver_package = 'pastix'


class rmhd2d_lu(rmhd2d):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        super().__init__(cfgfile)#rmhd2d_ppc, self
        
        
        OptDB = PETSc.Options()
        
        OptDB.setValue('ksp_monitor',  '')
        OptDB.setValue('snes_monitor', '')
        
#        OptDB.setValue('log_info',    '')
#        OptDB.setValue('log_summary', '')
        
        OptDB.setValue('snes_ls', 'basic')
#         OptDB.setValue('snes_ls', 'quadratic')

        OptDB.setValue('snes_rtol',   self.cfg['solver']['petsc_snes_rtol'])
        OptDB.setValue('snes_atol',   self.cfg['solver']['petsc_snes_atol'])
        OptDB.setValue('snes_stol',   self.cfg['solver']['petsc_snes_stol'])
        OptDB.setValue('snes_max_it', self.cfg['solver']['petsc_snes_max_iter'])
        
        OptDB.setValue('ksp_rtol',   self.cfg['solver']['petsc_ksp_rtol'])
        OptDB.setValue('ksp_atol',   self.cfg['solver']['petsc_ksp_atol'])
        OptDB.setValue('ksp_max_it', self.cfg['solver']['petsc_ksp_max_iter'])
        
#        OptDB.setValue('mat_superlu_dist_matinput', 'DISTRIBUTED')
#        OptDB.setValue('mat_superlu_dist_rowperm',  'NATURAL')
        OptDB.setValue('mat_superlu_dist_colperm',  'PARMETIS')
        OptDB.setValue('mat_superlu_dist_parsymbfact', 1)
        
        
        # create Jacobian, Function, and linear Matrix objects
        self.petsc_solver   = PETScSolver(self.da1, self.da4, self.nx, self.ny, self.ht, self.hx, self.hy)
        
        
        # initialise linear matrix
        self.M = self.da4.createMat()
        self.M.setOption(self.M.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.M.setUp()
        
        # initialise Jacobian
        self.Jac = self.da4.createMat()
        self.Jac.setOption(self.Jac.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.Jac.setUp()
        
        # initialise matrixfree Jacobian
        self.Jmf = PETSc.Mat().createPython([self.x.getSizes(), self.b.getSizes()], 
                                            context=self.petsc_solver,
                                            comm=PETSc.COMM_WORLD)
        self.Jmf.setUp()

        # create nonlinear solver
        self.snes = PETSc.SNES().create()
        self.snes.setFunction(self.petsc_solver.snes_function, self.f)
        self.snes.setJacobian(self.updateJacobian, self.Jmf, self.Jac)
        self.snes.setFromOptions()
        self.snes.getKSP().setType('preonly')
        self.snes.getKSP().getPC().setType('lu')
        self.snes.getKSP().getPC().setFactorSolverPackage(solver_package)

        # update solution history
        self.petsc_solver.update_previous(self.x)
        
        
    
    def __del__(self):
        self.snes.destroy()
        self.Jac.destroy()
        self.M.destroy()
    
    
    def updateJacobian(self, snes, X, J, P):
        self.petsc_solver.update_previous(X)
        self.petsc_solver.formMat(P)
    
    
    def run(self):
        
        for itime in range(1, self.nt+1):
            current_time = self.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
                print
                self.time.setValue(0, current_time)
            
            # calculate initial guess
            self.calculate_initial_guess(initial=itime==1)
            
            # update history
            self.petsc_solver.update_history()
            
            # copy initial guess to x
            x_arr = self.da4.getVecArray(self.x)
            x_arr[:,:,0] = self.da1.getVecArray(self.A)[:,:]
            x_arr[:,:,1] = self.da1.getVecArray(self.J)[:,:]
            x_arr[:,:,2] = self.da1.getVecArray(self.P)[:,:]
            x_arr[:,:,3] = self.da1.getVecArray(self.O)[:,:]
            
            # solve
            self.snes.solve(None, self.x)
            
            # compute function norm
            self.petsc_solver.update_previous(self.x)
            self.petsc_solver.function(self.f)
            norm = self.f.norm()
            
            # output some solver info
            if PETSc.COMM_WORLD.getRank() == 0:
                print()
                print("  Nonlin Solver:  %5i iterations,   funcnorm = %24.16E" % (self.snes.getIterationNumber(), norm) )
                print()
            
            if self.snes.getConvergedReason() < 0:
                if PETSc.COMM_WORLD.getRank() == 0:
                    print()
                    print("Solver not converging...   (Reason: %i)" % (self.snes.getConvergedReason()))
                    print()
           
           
            # save to hdf5 file
            if itime % self.nsave == 0 or itime == self.nt + 1:
                self.save_to_hdf5(itime)
            
        
    


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
