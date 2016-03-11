'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import numpy as np

import argparse
import time

from config import Config

from PETScDerivatives        import PETScDerivatives
from PETScOhmsLawArakawaJ1   import PETScOhmsLaw
from PETScVorticityArakawaJ1 import PETScVorticity
from PETScPoissonCFD2        import PETScPoisson


#solver_package = 'superlu_dist'
solver_package = 'mumps'


class petscMHD2D(object):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        # load run config file
        cfg = Config(cfgfile)
        
        # timestep setup
        self.ht    = cfg['grid']['ht']              # timestep size
        self.nt    = cfg['grid']['nt']              # number of timesteps
        self.nsave = cfg['io']['nsave']             # save only every nsave'th timestep
        
        # grid setup
        self.nx   = cfg['grid']['nx']                    # number of points in x
        self.ny   = cfg['grid']['ny']                    # number of points in y
        
        Lx   = cfg['grid']['Lx']                    # spatial domain in x
        x1   = cfg['grid']['x1']                    # 
        x2   = cfg['grid']['x2']                    # 
        
        Ly   = cfg['grid']['Ly']                    # spatial domain in y
        y1   = cfg['grid']['y1']                    # 
        y2   = cfg['grid']['y2']                    # 
        
        if x1 != x2:
            Lx = x2-x1
        else:
            x1 = 0.0
            x2 = Lx
        
        if y1 != y2:
            Ly = y2-y1
        else:
            y1 = 0.0
            y2 = Ly
        
        
        self.hx = Lx / self.nx                       # gridstep size in x
        self.hy = Ly / self.ny                       # gridstep size in y
        
        
        self.time = PETSc.Vec().createMPI(1, PETSc.DECIDE, comm=PETSc.COMM_WORLD)
        self.time.setName('t')
        
        if PETSc.COMM_WORLD.getRank() == 0:
            self.time.setValue(0, 0.0)
        
        
        OptDB = PETSc.Options()
        
        OptDB.setValue('snes_rtol',   cfg['solver']['petsc_snes_rtol'])
        OptDB.setValue('snes_atol',   cfg['solver']['petsc_snes_atol'])
        OptDB.setValue('snes_stol',   cfg['solver']['petsc_snes_stol'])
        OptDB.setValue('snes_max_it', cfg['solver']['petsc_snes_max_iter'])
        
        OptDB.setValue('ksp_rtol',   cfg['solver']['petsc_ksp_rtol'])
        OptDB.setValue('ksp_atol',   cfg['solver']['petsc_ksp_atol'])
        OptDB.setValue('ksp_max_it', cfg['solver']['petsc_ksp_max_iter'])
        
#         OptDB.setValue('ksp_monitor',  '')
#         OptDB.setValue('snes_monitor', '')
#         
#        OptDB.setValue('log_info',    '')
#        OptDB.setValue('log_summary', '')
        
        self.snes_rtol = cfg['solver']['petsc_snes_rtol']
        self.snes_atol = cfg['solver']['petsc_snes_atol']
        self.snes_max_iter = cfg['solver']['petsc_snes_max_iter']
        
        
        # create DA with single dof
        self.da1 = PETSc.DA().create(dim=2, dof=1,
                                    sizes=[self.nx, self.ny],
                                    proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                    boundary_type=('periodic', 'periodic'),
                                    stencil_width=2,
                                    stencil_type='box')
        
        
        # create DA (dof = 4 for A, J, P, O)
        self.da4 = PETSc.DA().create(dim=2, dof=4,
                                     sizes=[self.nx, self.ny],
                                     proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                     boundary_type=('periodic', 'periodic'),
                                     stencil_width=2,
                                     stencil_type='box')
        
        
        # create DA for x grid
        self.dax = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[self.nx],
                                    proc_sizes=[PETSc.DECIDE],
                                    boundary_type=('periodic'))
        
        # create DA for y grid
        self.day = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[self.ny],
                                    proc_sizes=[PETSc.DECIDE],
                                    boundary_type=('periodic'))
        
        
        # initialise grid
        self.da1.setUniformCoordinates(xmin=x1, xmax=x2,
                                       ymin=y1, ymax=y2)
        
        self.da4.setUniformCoordinates(xmin=x1, xmax=x2,
                                       ymin=y1, ymax=y2)
        
        self.dax.setUniformCoordinates(xmin=x1, xmax=x2)
        
        self.day.setUniformCoordinates(xmin=y1, xmax=y2)
        
        
        # create RHS vector
        self.Ab = self.da1.createGlobalVec()
        self.Ob = self.da1.createGlobalVec()
        self.Pb = self.da1.createGlobalVec()
        
        # create vectors for magnetic and velocity field
        self.A  = self.da1.createGlobalVec()        # magnetic vector potential A
        self.J  = self.da1.createGlobalVec()        # current density           J
        self.P  = self.da1.createGlobalVec()        # streaming function        psi
        self.O  = self.da1.createGlobalVec()        # vorticity                 omega

        self.Bx = self.da1.createGlobalVec()
        self.By = self.da1.createGlobalVec()
        self.Vx = self.da1.createGlobalVec()
        self.Vy = self.da1.createGlobalVec()
        
        # set variable names
        self.A.setName('A')
        self.J.setName('J')
        self.P.setName('P')
        self.O.setName('O')
        
        self.Bx.setName('Bx')
        self.By.setName('By')
        self.Vx.setName('Vx')
        self.Vy.setName('Vy')
        
        
        # create nullspace
        self.poisson_nullspace = PETSc.NullSpace().create(constant=True)
        
        # create jacobian and matrix objects
        self.petsc_vorticity = PETScVorticity(self.da1, self.nx, self.ny, self.ht, self.hx, self.hy)
        self.petsc_ohmslaw   = PETScOhmsLaw  (self.da1, self.nx, self.ny, self.ht, self.hx, self.hy)
        self.petsc_poisson   = PETScPoisson  (self.da1, self.nx, self.ny, self.hx, self.hy)
        
        # initialise vorticity matrix
        self.vorticity_matrix = self.da1.createMat()
        self.vorticity_matrix.setOption(self.vorticity_matrix.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.vorticity_matrix.setUp()
        
        # initialise Ohms's law matrix
        self.ohmslaw_matrix = self.da1.createMat()
        self.ohmslaw_matrix.setOption(self.ohmslaw_matrix.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.ohmslaw_matrix.setUp()
        
        # initialise Poisson matrix
        self.poisson_matrix = self.da1.createMat()
        self.poisson_matrix.setOption(self.poisson_matrix.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.poisson_matrix.setUp()
        self.poisson_matrix.setNullSpace(self.poisson_nullspace)
        
        # create nonlinear vorticity solver
        self.vorticity_snes = PETSc.SNES().create()
        self.vorticity_snes.setType('ksponly')
        self.vorticity_snes.setFunction(self.petsc_vorticity.snes_mult, self.Ob)
        self.vorticity_snes.setJacobian(self.update_vorticity_jacobian, self.vorticity_matrix)
        self.vorticity_snes.setFromOptions()
#         self.vorticity_snes.getKSP().setType('gmres')
        self.vorticity_snes.getKSP().setType('preonly')
        self.vorticity_snes.getKSP().getPC().setType('lu')
        self.vorticity_snes.getKSP().getPC().setFactorSolverPackage(solver_package)

        # create nonlinear Ohms's law solver
        self.ohmslaw_snes = PETSc.SNES().create()
        self.ohmslaw_snes.setType('ksponly')
        self.ohmslaw_snes.setFunction(self.petsc_ohmslaw.snes_mult, self.Ab)
        self.ohmslaw_snes.setJacobian(self.update_ohmslaw_jacobian, self.ohmslaw_matrix)
        self.ohmslaw_snes.setFromOptions()
#         self.ohmslaw_snes.getKSP().setType('gmres')
        self.ohmslaw_snes.getKSP().setType('preonly')
        self.ohmslaw_snes.getKSP().getPC().setType('lu')
        self.ohmslaw_snes.getKSP().getPC().setFactorSolverPackage(solver_package)

        # create linear Poisson solver
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.poisson_matrix)
        self.poisson_ksp.setType('preonly')
        self.poisson_ksp.getPC().setType('lu')
        self.poisson_ksp.getPC().setFactorSolverPackage(solver_package)
#         self.poisson_ksp.setNullSpace(self.poisson_nullspace)
        self.petsc_poisson.formMat(self.poisson_matrix)
        
        
        # create derivatives object
        self.derivatives = PETScDerivatives(self.da1, self.nx, self.ny, self.ht, self.hx, self.hy)
        
        # get coordinate vectors
        coords_x = self.dax.getCoordinates()
        coords_y = self.day.getCoordinates()
         
        # save x coordinate arrays
        scatter, xVec = PETSc.Scatter.toAll(coords_x)
        
        scatter.begin(coords_x, xVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        scatter.end  (coords_x, xVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
          
        xGrid = xVec.getValues(range(0, self.nx)).copy()
          
        scatter.destroy()
        xVec.destroy()
          
        # save y coordinate arrays
        scatter, yVec = PETSc.Scatter.toAll(coords_y)
        
        scatter.begin(coords_y, yVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        scatter.end  (coords_y, yVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
                    
        yGrid = yVec.getValues(range(0, self.ny)).copy()
          
        scatter.destroy()
        yVec.destroy()

        # set initial data
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        A_arr = self.da1.getVecArray(self.A)
        P_arr = self.da1.getVecArray(self.P)
        
        init_data = __import__("runs." + cfg['initial_data']['python'], globals(), locals(), ['magnetic_A', 'velocity_P'], 0)
        
        for i in range(xs, xe):
            for j in range(ys, ye):
                A_arr[i,j] = init_data.magnetic_A(xGrid[i], yGrid[j], Lx, Ly)
                P_arr[i,j] = init_data.velocity_P(xGrid[i], yGrid[j], Lx, Ly)
        
        # Fourier Filtering
        self.nfourier = cfg['initial_data']['nfourier']
          
        if self.nfourier >= 0:
            # obtain whole A vector everywhere
            scatter, Aglobal = PETSc.Scatter.toAll(self.A)
            
            scatter.begin(self.A, Aglobal, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
            scatter.end  (self.A, Aglobal, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
            
            petsc_indices = self.da1.getAO().app2petsc(np.arange(self.nx*self.ny, dtype=np.int32))
            
            Ainit = Aglobal.getValues(petsc_indices).copy().reshape((self.ny, self.nx))
            
            scatter.destroy()
            Aglobal.destroy()
            
            # compute FFT, cut, compute inverse FFT
            from scipy.fftpack import rfft, irfft
            
            Afft = rfft(Ainit, axis=1)
            
            Afft[:,0] = 0.
            Afft[:,self.nfourier+1:] = 0.
            
            A_arr = self.da1.getVecArray(self.A)
            A_arr[:,:] = irfft(Afft).T[xs:xe, ys:ye] 
            
        
        # compute current and vorticity
        self.derivatives.laplace_vec(self.A, self.J, -1.)
        self.derivatives.laplace_vec(self.P, self.O, -1.)
        
        J_arr = self.da1.getVecArray(self.J)
        O_arr = self.da1.getVecArray(self.O)
        
        # add perturbations
        for i in range(xs, xe):
            for j in range(ys, ye):
                J_arr[i,j] += init_data.current_perturbation(  xGrid[i], yGrid[j], Lx, Ly)
                O_arr[i,j] += init_data.vorticity_perturbation(xGrid[i], yGrid[j], Lx, Ly)
        
        
        # create HDF5 output file
        self.hdf5_viewer = PETSc.ViewerHDF5().create(cfg['io']['hdf5_output'],
                                          mode=PETSc.Viewer.Mode.WRITE,
                                          comm=PETSc.COMM_WORLD)
        
        self.hdf5_viewer.pushGroup("/")
        
        
        # write grid data to hdf5 file
        coords_x = self.dax.getCoordinates()
        coords_y = self.day.getCoordinates()
        
        coords_x.setName('x')
        coords_y.setName('y')

        self.hdf5_viewer(coords_x)
        self.hdf5_viewer(coords_y)
        
        
        # write initial data to hdf5 file
        self.save_to_hdf5(0)
        
        
    
    def __del__(self):
        self.hdf5_viewer.destroy()
        self.vorticity_snes.destroy()
        self.ohmslaw_snes.destroy()
        self.poisson_ksp.destroy()
        
    
    def update_vorticity_jacobian(self, snes, X, J, P):
        self.petsc_vorticity.formMat(P)
    
    def update_ohmslaw_jacobian(self, snes, X, J, P):
        self.petsc_ohmslaw.formMat(P)
    
    
    def run(self):
        
        for itime in range(1, self.nt+1):
            current_time = self.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
                print
                self.time.setValue(0, current_time)
            
                # update history in vorticity solver
                self.petsc_vorticity.update_history(self.O, self.P, self.A, self.J)
                
                # compute initial norm
                self.petsc_vorticity.function(self.O, self.Ob)
                norm1 = self.Ob.norm()
                
                j = 0
                while True:
                    j += 1
                    
                    # solve for vorticity
                    self.vorticity_snes.solve(None, self.O)
                    
                    # solve for streaming function
                    self.poisson_ksp.solve(self.O, self.P)
                    
                    # compute norm
                    self.petsc_vorticity.update_streaming_function(self.P)
                    self.petsc_vorticity.function(self.O, self.Ob)
                    norm2 = norm1
                    norm1 = self.Ob.norm()
                    
                    if PETSc.COMM_WORLD.getRank() == 0:
                        print("   Vorticity Solver:  %5i iterations,   funcnorm = %24.16E" % (j, norm1) )
                    
                    if norm1 < self.snes_atol or j >= self.snes_max_iter or np.abs(norm1 - norm2) < self.snes_rtol:
                        break
            
                # update history in Ohm's law solver
                self.petsc_ohmslaw.update_history(self.A, self.P)
                
                # solve for magnetic vector potential
                self.ohmslaw_snes.solve(None, self.A)
                
                # update current density
                self.derivatives.laplace_vec(self.A, self.J, -1.)
                

            # save to hdf5 file
            self.save_to_hdf5(itime)
            
    
    def save_to_hdf5(self, timestep):
        if timestep % self.nsave == 0 or timestep == self.nt + 1:
            
            # calculate B and V field
            self.derivatives.dy(self.A, self.Bx, +1.)
            self.derivatives.dx(self.A, self.By, -1.)
            self.derivatives.dy(self.P, self.Vx, +1.)
            self.derivatives.dx(self.P, self.Vy, -1.)
            
            # save timestep
            self.hdf5_viewer.setTimestep(timestep)
            self.hdf5_viewer(self.time)
            
            self.hdf5_viewer(self.A)
            self.hdf5_viewer(self.J)
            self.hdf5_viewer(self.P)
            self.hdf5_viewer(self.O)
            
            self.hdf5_viewer(self.Bx)
            self.hdf5_viewer(self.By)
            self.hdf5_viewer(self.Vx)
            self.hdf5_viewer(self.Vy)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc MHD Solver in 2D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscMHD2D(args.runfile)
    petscvp.run()
    
