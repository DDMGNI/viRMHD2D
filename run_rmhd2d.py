'''
Created on 21.03.2016

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import h5py
import numpy as np

import argparse, datetime, time
import pstats, cProfile

from config import Config

from PETScDerivatives                    import PETScDerivatives
from PETScPoissonCFD2                    import PETScPoisson
from PETScNonlinearSolverArakawaJ1CFD2   import PETScSolver
from PETScPreconditionerArakawaJ1CFD2    import PETScPreconditioner
# from PETScPreconditionerArakawaJ1CFD2Vec import PETScPreconditioner


solver_package = 'superlu_dist'
# solver_package = 'mumps'
# solver_package = 'pastix'


class rmhd2d(object):
    '''
    PETSc/Python Reduced MHD Solver in 2D.
    '''
        
    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        # solver mode
        self.mode = "none"
        
        # set run id to timestamp
        self.run_id = datetime.datetime.fromtimestamp(time.time()).strftime("%y%m%d%H%M%S")
        
        # load run config file
        self.cfg = Config(cfgfile)
        
        # timestep setup
        self.ht    = self.cfg['grid']['ht']              # timestep size
        self.nt    = self.cfg['grid']['nt']              # number of timesteps
        self.nsave = self.cfg['io']['nsave']             # save only every nsave'th timestep
        
        # grid setup
        self.nx = self.cfg['grid']['nx']                 # number of points in x
        self.ny = self.cfg['grid']['ny']                 # number of points in y
        
        self.Lx = self.cfg['grid']['Lx']                 # spatial domain in x
        x1      = self.cfg['grid']['x1']                 # 
        x2      = self.cfg['grid']['x2']                 # 
        
        self.Ly = self.cfg['grid']['Ly']                 # spatial domain in y
        y1      = self.cfg['grid']['y1']                 # 
        y2      = self.cfg['grid']['y2']                 # 
        
        self.hx = self.cfg['grid']['hx']                 # gridstep size in x
        self.hy = self.cfg['grid']['hy']                 # gridstep size in y
        
        # create time vector
        self.time = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
        self.time.setName('t')
        
        # electron skin depth
        self.de = self.cfg['initial_data']['skin_depth']
        
        # set global tolerance
        self.tolerance = self.cfg['solver']['petsc_snes_atol'] * self.nx * self.ny
        
        # create DA with single dof
        self.da1 = PETSc.DA().create(dim=2, dof=1,
                                    sizes=[self.nx, self.ny],
                                    proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                    boundary_type=('periodic', 'periodic'),
                                    stencil_width=1,
                                    stencil_type='box')
        
        # create DA (dof = 4 for A, J, P, O)
        self.da4 = PETSc.DA().create(dim=2, dof=4,
                                     sizes=[self.nx, self.ny],
                                     proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                     boundary_type=('periodic', 'periodic'),
                                     stencil_width=1,
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
        self.dx = self.da4.createGlobalVec()
        self.dy = self.da4.createGlobalVec()
        self.x  = self.da4.createGlobalVec()
        self.b  = self.da4.createGlobalVec()
        self.f  = self.da4.createGlobalVec()
        self.Pb = self.da1.createGlobalVec()

        self.FA = self.da1.createGlobalVec()
        self.FJ = self.da1.createGlobalVec()
        self.FP = self.da1.createGlobalVec()
        self.FO = self.da1.createGlobalVec()
        
        # create initial guess vectors
        self.igFA = self.da1.createGlobalVec()
        self.igFO1= self.da1.createGlobalVec()
        self.igFO2= self.da1.createGlobalVec()
        
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
        
        x0_arr[:,:,2] = 1.
        self.x0.assemble()
        self.x0.normalize()
        
        self.solver_nullspace  = PETSc.NullSpace().create(constant=False, vectors=(self.x0,))
        self.poisson_nullspace = PETSc.NullSpace().create(constant=True)
        
        
#         # place holder for Poisson solver
#         self.poisson_ksp = None
        
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
        
        init_data = __import__("runs." + self.cfg['initial_data']['python'], globals(), locals(), ['magnetic_A', 'velocity_P'], 0)
        
        for i in range(xs, xe):
            for j in range(ys, ye):
                A_arr[i,j] = init_data.magnetic_A(xGrid[i], yGrid[j], Lx, Ly)
                P_arr[i,j] = init_data.velocity_P(xGrid[i], yGrid[j], Lx, Ly)
        
        # Fourier Filtering
        self.nfourier = self.cfg['initial_data']['nfourier']
          
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
            
#             Afft[:,0] = 0.
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
        
        
        # initialise Poisson matrix
        self.Pm = self.da1.createMat()
        self.Pm.setOption(self.Pm.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.Pm.setUp()
        self.Pm.setNullSpace(self.poisson_nullspace)
        
        # create Poisson solver object
        self.petsc_poisson  = PETScPoisson(self.da1, self.nx, self.ny, self.hx, self.hy)
        
        # setup linear Poisson solver
        self.poisson_ksp = PETSc.KSP().create()
#         self.poisson_ksp.setDM(self.da1)
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.Pm)
#         self.poisson_ksp.setTolerances(rtol=1E-15, atol=1E-16)
        self.poisson_ksp.setTolerances(rtol=self.cfg['solver']['poisson_ksp_rtol'],
                                       atol=self.cfg['solver']['poisson_ksp_atol'],
                                       max_it=self.cfg['solver']['poisson_ksp_max_iter'])
        self.poisson_ksp.setType('cg')
        self.poisson_ksp.getPC().setType('hypre')
        
        self.petsc_poisson.formMat(self.Pm)
        
        # solve for consistent initial A
        self.A.set(0.)
        self.petsc_poisson.formRHS(self.J, self.Pb)
        self.poisson_nullspace.remove(self.Pb)
        self.poisson_ksp.solve(self.Pb, self.A)
        
        # solve for consistent initial psi
        self.P.set(0.)
        self.petsc_poisson.formRHS(self.O, self.Pb)
        self.poisson_nullspace.remove(self.Pb)
        self.poisson_ksp.solve(self.Pb, self.P)
        
        # copy initial data vectors to x
        x_arr = self.da4.getVecArray(self.x)
        x_arr[:,:,0] = self.da1.getVecArray(self.A)[:,:]
        x_arr[:,:,1] = self.da1.getVecArray(self.J)[:,:]
        x_arr[:,:,2] = self.da1.getVecArray(self.P)[:,:]
        x_arr[:,:,3] = self.da1.getVecArray(self.O)[:,:]
        
        # create HDF5 output file
        hdf5out = h5py.File(self.cfg['io']['hdf5_output'], "w", driver="mpio", comm=PETSc.COMM_WORLD.tompi4py())
        
#         hdf5_params = hdf5out.create_group("parameters")
#         hdf5_params.attrs["run_id"] = self.run_id
#         
#         for cfg_group in self.cfg:
#             hdf5_group = hdf5_params.create_group(cfg_group)
#             
#             for cfg_item in self.cfg[cfg_group]:
#                 hdf5_group.attrs[cfg_item] = self.cfg[cfg_group][cfg_item]
#         
#         python_file = open("runs/" + self.cfg['initial_data']['python'] + ".py", 'r')
#         
#         hdf5_params["initial_data"].attrs["python_file"] = python_file.read()
#         
#         python_file.close()
        
        hdf5out.attrs["run_id"] = self.run_id
        
        for cfg_group in self.cfg:
            for cfg_item in self.cfg[cfg_group]:
                if self.cfg[cfg_group][cfg_item] != None:
                    value = self.cfg[cfg_group][cfg_item]
                else:
                    value = ""
                    
                print(cfg_group + "." + cfg_item, value)
                hdf5out.attrs[cfg_group + "." + cfg_item] = value
        
        python_file = open("runs/" + self.cfg['initial_data']['python'] + ".py", 'r')
        
        hdf5out.attrs["initial_data.python_file"] = python_file.read()
        
        python_file.close()
        
        hdf5out.close()
        
        # create HDF5 viewer
        self.hdf5_viewer = PETSc.ViewerHDF5().create(self.cfg['io']['hdf5_output'],
                                          mode=PETSc.Viewer.Mode.APPEND,
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
        self.poisson_ksp.destroy()
        self.Pm.destroy()
        self.hdf5_viewer.destroy()


    def run(self):
        raise NotImplementedError


    def calculate_initial_guess(self, initial=False):
        
        # set some variables for hermite extrapolation
        t0 = 0.
        t1 = 1.
        t  = 2.
        
        a0 = 2./(t0-t1)
        a1 = 2./(t1-t0)
        
        b0 = 1./(t0-t1)**2
        b1 = 1./(t1-t0)**2
        
        d0 = 1./(t-t0)
        d1 = 1./(t-t1)
        
        e0 = d0*b0
        e1 = d1*b1
        
        self.hermite_x0 = e0*(d0-a0)
        self.hermite_x1 = e1*(d1-a1)
        
        self.hermite_f0 = e0*self.ht
        self.hermite_f1 = e1*self.ht
        
        self.hermite_den = 1. / (self.hermite_x0 + self.hermite_x1)
        
        
        self.petsc_solver.Ap.copy(self.A)
        self.petsc_solver.Op.copy(self.O)
        
        if initial:
            self.derivatives.arakawa_vec(self.petsc_solver.Ap, self.petsc_solver.Pp, self.igFA)
            self.derivatives.arakawa_vec(self.petsc_solver.Op, self.petsc_solver.Pp, self.igFO1)
            self.derivatives.arakawa_vec(self.petsc_solver.Ap, self.petsc_solver.Jp, self.igFO2)
            
            self.A.axpy(0.5*self.ht, self.igFA)
            self.O.axpy(0.5*self.ht, self.igFO1)
            self.O.axpy(0.5*self.ht, self.igFO2)
            
            self.derivatives.laplace_vec(self.A, self.J, -1.)
            
            self.P.set(0.)
            self.O.copy(self.Pb)
            self.poisson_nullspace.remove(self.Pb)
            self.poisson_ksp.solve(self.Pb, self.P)
            
            self.derivatives.arakawa_vec(self.A, self.P, self.igFA)
            self.derivatives.arakawa_vec(self.O, self.P, self.igFO1)
            self.derivatives.arakawa_vec(self.A, self.J, self.igFO2)
            
            self.petsc_solver.Ap.copy(self.A)
            self.petsc_solver.Op.copy(self.O)
            
            self.A.axpy(self.ht, self.igFA)
            self.O.axpy(self.ht, self.igFO1)
            self.O.axpy(self.ht, self.igFO2)

            self.derivatives.arakawa_vec(self.petsc_solver.Ap, self.petsc_solver.Pp, self.igFA)
            self.derivatives.arakawa_vec(self.petsc_solver.Op, self.petsc_solver.Pp, self.igFO1)
            self.derivatives.arakawa_vec(self.petsc_solver.Ap, self.petsc_solver.Jp, self.igFO2)
            
        else:
            self.A.set(0.)
            self.O.set(0.)
    
            self.A.axpy(self.hermite_x0, self.petsc_solver.Ah)
            self.A.axpy(self.hermite_f0, self.igFA)
            
            self.O.axpy(self.hermite_x0, self.petsc_solver.Oh)
            self.O.axpy(self.hermite_f0, self.igFO1)
            self.O.axpy(self.hermite_f0, self.igFO2)
            
            self.derivatives.arakawa_vec(self.petsc_solver.Ap, self.petsc_solver.Pp, self.igFA)
            self.derivatives.arakawa_vec(self.petsc_solver.Op, self.petsc_solver.Pp, self.igFO1)
            self.derivatives.arakawa_vec(self.petsc_solver.Ap, self.petsc_solver.Jp, self.igFO2)
            
            self.A.axpy(self.hermite_x1, self.petsc_solver.Ap)
            self.A.axpy(self.hermite_f1, self.igFA)
            
            self.O.axpy(self.hermite_x1, self.petsc_solver.Op)
            self.O.axpy(self.hermite_f1, self.igFO1)
            self.O.axpy(self.hermite_f1, self.igFO2)
            
            self.A.scale(self.hermite_den)
            self.O.scale(self.hermite_den)
        
        self.derivatives.laplace_vec(self.A, self.J, -1.)
        
        self.P.set(0.)
        self.O.copy(self.Pb)
        self.poisson_nullspace.remove(self.Pb)
        self.poisson_ksp.solve(self.Pb, self.P)
        
        
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
        
        
        
        self.hdf5_viewer(self.A)
        self.hdf5_viewer(self.J)
        self.hdf5_viewer(self.P)
        self.hdf5_viewer(self.O)
        
        self.hdf5_viewer(self.Bx)
        self.hdf5_viewer(self.By)
        self.hdf5_viewer(self.Vx)
        self.hdf5_viewer(self.Vy)
        if timestep % self.nsave == 0:
            if PETSc.COMM_WORLD.getRank() == 0:
                self.time.setValue(0, self.ht*timestep)
            
            # save timestep
            self.hdf5_viewer.setTimestep(timestep // self.nsave)
            self.hdf5_viewer(self.time)
        

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
                        self.Jac.function(ex, dJ)
                        
                        dJ_arr = self.da4.getVecArray(dJ)
                        Jx_arr = self.da4.getVecArray(Jx)
                        Jx_arr[ix, iy, tfield] = dJ_arr[ix, iy, tfield]
                        
                        
                        # compute F(x - eps ex)
                        self.x.copy(dx)
                        dx_arr = self.da4.getVecArray(dx)
                        dx_arr[(ix+sx) % self.nx, (iy+sy) % self.ny, ifield] -= eps
                        
                        self.petsc_solver.function(dx, Fxm)
                        
                        
                        # compute F(x + eps ex)
                        self.x.copy(dx)
                        dx_arr = self.da4.getVecArray(dx)
                        dx_arr[(ix+sx) % self.nx, (iy+sy) % self.ny, ifield] += eps
                        
                        self.petsc_solver.function(dx, Fxp)
                        
                        
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
    
#     if OptDB.hasName("h"):
#         
#         sys.exit(0)
    
    
    runfile = OptDB.getString('c')
    mode    = OptDB.getString('m')
    
    if mode == "ppc":
        # physics based preconditioner
        from run_rmhd2d_ppc import rmhd2d_ppc
        petscvp = rmhd2d_ppc(runfile)
    elif mode == "split":
        # split solver with physics based preconditioner
        from run_rmhd2d_split import rmhd2d_split
        petscvp = rmhd2d_split(runfile)
    elif mode == "asm":
        # additive schwarz preconditioner
        from run_rmhd2d_asm import rmhd2d_asm
        petscvp = rmhd2d_asm(runfile)
    else:
        # direct solver (lu decomposition)
        from run_rmhd2d_lu import rmhd2d_lu
        petscvp = rmhd2d_lu(runfile)

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
