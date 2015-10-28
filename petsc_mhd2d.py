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

from PETScDerivatives       import PETScDerivatives
from PETScSimplePoisson     import PETScPoisson
# from PETScSimpleNLMatrix    import PETScMatrix
from PETScSimpleNLSolver    import PETScSolver


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
        self.nx    = cfg['grid']['nx']              # number of points in x
        self.ny    = cfg['grid']['ny']              # number of points in y
        
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
        
        OptDB.setValue('ksp_monitor',  '')
        OptDB.setValue('snes_monitor', '')
        
#        OptDB.setValue('log_info',    '')
#        OptDB.setValue('log_summary', '')
        
        
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
        
        
        # create solution and RHS vector
        self.x  = self.da4.createGlobalVec()
        self.b  = self.da4.createGlobalVec()
        self.f  = self.da4.createGlobalVec()
        self.Pb = self.da1.createGlobalVec()
        
        #  nullspace vectors
        self.x0 = self.da4.createGlobalVec()
        self.P0 = self.da1.createGlobalVec()
        
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
        
        
        # initialise nullspace basis vectors
        self.x0.set(0.)
        x0_arr = self.da4.getVecArray(self.x0)[...]
        
        x0_arr[:, :, 2] = 1.
        self.x0.normalize()
        
        self.solver_nullspace  = PETSc.NullSpace().create(constant=False, vectors=(self.x0,))
        self.poisson_nullspace = PETSc.NullSpace().create(constant=True)
        
        
        # create Jacobian, Function, and linear Matrix objects
        self.petsc_solver   = PETScSolver(self.da1, self.da4, self.nx, self.ny, self.ht, self.hx, self.hy)
#         self.petsc_matrix   = PETScMatrix(self.da1, self.da4, self.nx, self.ny, self.ht, self.hx, self.hy)
        self.petsc_poisson  = PETScPoisson(self.da1, self.nx, self.ny, self.hx, self.hy)
        
        
        # initialise Poisson matrix
        self.Pm = self.da1.createMat()
        self.Pm.setOption(self.Pm.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.Pm.setUp()
        self.Pm.setNullSpace(self.poisson_nullspace)
        
        # initialise linear matrix
        self.M = self.da4.createMat()
        self.M.setOption(self.M.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.M.setUp()
        
        # initialise Jacobian
        self.Jac = self.da4.createMat()
        self.Jac.setOption(self.Jac.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.Jac.setUp()
        
        # create nonlinear solver
        self.snes = PETSc.SNES().create()
        self.snes.setFunction(self.petsc_solver.snes_mult, self.f)
        self.snes.setJacobian(self.updateJacobian, self.Jac)
        self.snes.setFromOptions()
#         self.snes.getKSP().setType('gmres')
        self.snes.getKSP().setType('preonly')
#         self.snes.getKSP().setNullSpace(self.solver_nullspace)
        self.snes.getKSP().getPC().setType('lu')
        self.snes.getKSP().getPC().setFactorSolverPackage(solver_package)

        # place holder for Poisson solver
        self.poisson_ksp = None
        
        # create derivatives object
        self.derivatives = PETScDerivatives(self.da1, self.nx, self.ny, self.ht, self.hx, self.hy)
        
        # get coordinate vectors
        coords_x = self.dax.getCoordinates()
        coords_y = self.day.getCoordinates()
         
        # save x coordinate arrays
        scatter, xVec = PETSc.Scatter.toAll(coords_x)
        
        scatter.begin(coords_x, xVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        scatter.end  (coords_x, xVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
          
        xGrid = xVec.getValues(range(self.nx)).copy()
          
        scatter.destroy()
        xVec.destroy()
          
        # save y coordinate arrays
        scatter, yVec = PETSc.Scatter.toAll(coords_y)
        
        scatter.begin(coords_y, yVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        scatter.end  (coords_y, yVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
                    
        yGrid = yVec.getValues(range(self.ny)).copy()
          
        scatter.destroy()
        yVec.destroy()

        # set initial data
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        x_arr = self.da4.getVecArray(self.x)
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
        
        
        # solve for consistent initial A
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.Pm)
        self.poisson_ksp.setType('preonly')
        self.poisson_ksp.getPC().setType('lu')
        self.poisson_ksp.getPC().setFactorSolverPackage(solver_package)
#         self.poisson_ksp.setNullSpace(self.poisson_nullspace)
        
        self.petsc_poisson.formMat(self.Pm)
        self.petsc_poisson.formRHS(self.J, self.Pb)
        self.poisson_ksp.solve(self.Pb, self.A)
        
        del self.poisson_ksp
        
        
        # solve for consistent initial psi
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.Pm)
        self.poisson_ksp.setType('preonly')
        self.poisson_ksp.getPC().setType('lu')
        self.poisson_ksp.getPC().setFactorSolverPackage(solver_package)
#         self.poisson_ksp.setNullSpace(self.poisson_nullspace)
        
        self.petsc_poisson.formMat(self.Pm)
        self.petsc_poisson.formRHS(self.O, self.Pb)
        self.poisson_ksp.solve(self.Pb, self.P)
        
        del self.poisson_ksp
        
        
        # copy initial data vectors to x
        x_arr = self.da4.getVecArray(self.x)
        A_arr = self.da1.getVecArray(self.A)
        J_arr = self.da1.getVecArray(self.J)
        P_arr = self.da1.getVecArray(self.P)
        O_arr = self.da1.getVecArray(self.O)
        
        x_arr[xs:xe, ys:ye, 0] = A_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 1] = J_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 2] = P_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 3] = O_arr[xs:xe, ys:ye]
        
        
        # update solution history
        self.petsc_solver.update_history(self.x)
#         self.petsc_matrix.update_history(self.x)
        
        
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
#         self.poisson_ksp.destroy()
        self.snes.destroy()
        self.Jac.destroy()
        self.M.destroy()
        self.Pm.destroy()
    
    
    def updateJacobian(self, snes, X, J, P):
        self.petsc_solver.update_previous(X)
        self.petsc_solver.formMat(J)
#         J.setNullSpace(self.solver_nullspace)
    
    
    def run(self):
        
        for itime in range(1, self.nt+1):
            current_time = self.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
                print
                self.time.setValue(0, current_time)
            
            # calculate initial guess
#            self.calculate_initial_guess()
            
            # solve
            self.snes.solve(None, self.x)
            
            # compute function norm
            self.petsc_solver.mult(self.x, self.f)
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
           
           
            # update history
            self.petsc_solver.update_history(self.x)
#             self.petsc_matrix.update_history(self.x)
            
            # save to hdf5 file
#            if itime % self.nsave == 0 or itime == self.grid.nt + 1:
            self.save_to_hdf5(itime)
            
        
#     def calculate_initial_guess(self):
#         self.poisson_ksp = PETSc.KSP().create()
#         self.poisson_ksp.setFromOptions()
#         self.poisson_ksp.setOperators(self.M)
#         self.poisson_ksp.setType('preonly')
#         self.poisson_ksp.getPC().setType('lu')
#         self.poisson_ksp.getPC().setFactorSolverPackage(solver_package)
#     
#         # build matrix
#         self.petsc_matrix.formMat(self.M)
#         
# #        mat_viewer = PETSc.Viewer().createDraw(size=(800,800), comm=PETSc.COMM_WORLD)
# #        mat_viewer(self.M)
# #        input("Press Enter")
#         
#         # build RHS
#         self.b.set(0.)
# #        self.petsc_matrix.formRHS(self.b)
#         
#         # solve
#         self.poisson_ksp.solve(self.b, self.x)
        
    
    
    def save_to_hdf5(self, timestep):
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        # copy solution to A, J, psi, omega vectors
        x_arr = self.da4.getVecArray(self.x)
        A_arr = self.da1.getVecArray(self.A)
        J_arr = self.da1.getVecArray(self.J)
        P_arr = self.da1.getVecArray(self.P)
        O_arr = self.da1.getVecArray(self.O)

        A_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 0]
        J_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 1]
        P_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 2]
        O_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 3]
        
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
            
        

    def check_jacobian(self):
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        eps = 1.E-7
        
        # calculate initial guess
#        self.calculate_initial_guess()
        
        # update previous iteration
        self.petsc_solver.update_previous(self.x)
        
        # calculate jacobian
        self.petsc_solver.formMat(self.Jac)
        
        # create working vectors
        Jx  = self.da4.createGlobalVec()
        dJ  = self.da4.createGlobalVec()
        ex  = self.da4.createGlobalVec()
        dx  = self.da4.createGlobalVec()
        dF  = self.da4.createGlobalVec()
        Fxm = self.da4.createGlobalVec()
        Fxp = self.da4.createGlobalVec()
        
        
#         sx = -2
#         sx = -1
        sx =  0
#         sx = +1
#         sx = +2

#         sy = -1
        sy =  0
#         sy = +1
        
        nfield=4
        
        for ifield in range(0, nfield):
            for ix in range(xs, xe):
                for iy in range(ys, ye):
                    for tfield in range(0, nfield):
                        
                        # compute ex
                        ex_arr = self.da4.getVecArray(ex)
                        ex_arr[:] = 0.
                        ex_arr[(ix+sx) % self.nx, (iy+sy) % self.ny, ifield] = 1.
                        
                        
                        # compute J.e
                        self.Jac.mult(ex, dJ)
                        
                        dJ_arr = self.da4.getVecArray(dJ)
                        Jx_arr = self.da4.getVecArray(Jx)
                        Jx_arr[ix, iy, tfield] = dJ_arr[ix, iy, tfield]
                        
                        
                        # compute F(x - eps ex)
                        self.x.copy(dx)
                        dx_arr = self.da4.getVecArray(dx)
                        dx_arr[(ix+sx) % self.nx, (iy+sy) % self.ny, ifield] -= eps
                        
                        self.petsc_solver.mult(dx, Fxm)
                        
                        
                        # compute F(x + eps ex)
                        self.x.copy(dx)
                        dx_arr = self.da4.getVecArray(dx)
                        dx_arr[(ix+sx) % self.nx, (iy+sy) % self.ny, ifield] += eps
                        
                        self.petsc_solver.mult(dx, Fxp)
                        
                        
                        # compute dF = [F(x + eps ex) - F(x - eps ex)] / (2 eps)
                        Fxm_arr = self.da4.getVecArray(Fxm)
                        Fxp_arr = self.da4.getVecArray(Fxp)
                        dF_arr  = self.da4.getVecArray(dF)
                        
                        dF_arr[ix, iy, tfield] = ( Fxp_arr[ix, iy, tfield] - Fxm_arr[ix, iy, tfield] ) / (2. * eps)
                        
            
            diff = np.zeros(nfield)
            
            for tfield in range(0,nfield):
#                print()
#                print("Fields: (%5i, %5i)" % (ifield, tfield))
#                print()
                
                Jx_arr = self.da4.getVecArray(Jx)[...][:, :, tfield]
                dF_arr = self.da4.getVecArray(dF)[...][:, :, tfield]
                
                
#                 print("Jacobian:")
#                 print(Jx_arr)
#                 print()
#                 
#                 print("[F(x+dx) - F(x-dx)] / [2 eps]:")
#                 print(dF_arr)
#                 print()
#                
#                print("Difference:")
#                print(Jx_arr - dF_arr)
#                print()
                
                
#                if ifield == 3 and tfield == 2:
#                    print("Jacobian:")
#                    print(Jx_arr)
#                    print()
#                    
#                    print("[F(x+dx) - F(x-dx)] / [2 eps]:")
#                    print(dF_arr)
#                    print()
                
                
                diff[tfield] = (Jx_arr - dF_arr).max()
            
            print()
        
            for tfield in range(0,nfield):
                print("max(difference)[field=%i, equation=%i] = %16.8E" % ( ifield, tfield, diff[tfield] ))
            
            print()
    


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc MHD Solver in 2D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscMHD2D(args.runfile)
    petscvp.run()
#     petscvp.check_jacobian()
    
