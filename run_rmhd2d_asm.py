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


class rmhd2d_asm(rmhd2d):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        super().__init__(cfgfile, mode="asm")
        
        
        OptDB = PETSc.Options()
        
        OptDB.setValue('ksp_monitor',  '')
        OptDB.setValue('snes_monitor', '')
        
#        OptDB.setValue('log_info',    '')
#        OptDB.setValue('log_summary', '')
        
#         OptDB.setValue('snes_ls', 'basic')
        OptDB.setValue('snes_ls', 'quadratic')

        OptDB.setValue('pc_asm_type',  'restrict')
        OptDB.setValue('pc_asm_overlap', 3)
        OptDB.setValue('sub_ksp_type', 'preonly')
        OptDB.setValue('sub_pc_type', 'lu')
        OptDB.setValue('sub_pc_factor_mat_solver_package', self.solver_package)
        
        OptDB.setValue('snes_rtol',   self.cfg['solver']['petsc_snes_rtol'])
        OptDB.setValue('snes_atol',   self.cfg['solver']['petsc_snes_atol'])
        OptDB.setValue('snes_stol',   self.cfg['solver']['petsc_snes_stol'])
        OptDB.setValue('snes_max_it', self.cfg['solver']['petsc_snes_max_iter'])
        
        OptDB.setValue('ksp_rtol',   self.cfg['solver']['petsc_ksp_rtol'])
        OptDB.setValue('ksp_atol',   self.cfg['solver']['petsc_ksp_atol'])
        OptDB.setValue('ksp_max_it', self.cfg['solver']['petsc_ksp_max_iter'])
        
        OptDB.setValue('pc_type', 'hypre')
        OptDB.setValue('pc_hypre_type', 'boomeramg')
        OptDB.setValue('pc_hypre_boomeramg_max_iter', 2)
        
        
        # create Jacobian, Function, and linear Matrix objects
        self.petsc_solver   = PETScSolver(self.da1, self.da4, self.nx, self.ny, self.ht, self.hx, self.hy, self.de)
        
        
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
        self.snes.getKSP().setType('gmres')
        self.snes.getKSP().getPC().setType('asm')

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
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                print
            
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
            self.save_to_hdf5(itime)
            

