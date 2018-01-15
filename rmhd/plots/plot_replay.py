'''
Created on Jul 02, 2012

@author: mkraus
'''

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter


class PlotMHD2D(object):
    '''
    classdocs
    '''

    def __init__(self, diagnostics, nTime=0, nPlot=1, write=False):
        '''
        Constructor
        '''
        
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        
        self.write = write
        self.prefix = 'viRMHD2D_'
        
        self.nrows = 2
        self.ncols = 4
        
        if nTime > 0 and nTime < diagnostics.nt:
            self.nTime = nTime
        else:
            self.nTime = diagnostics.nt
        
        self.iTime = 0
        self.nPlot = nPlot
        
        self.diagnostics = diagnostics
        
        
        self.k_energy    = np.zeros_like(diagnostics.tGrid)
        self.m_energy    = np.zeros_like(diagnostics.tGrid)
        self.energy      = np.zeros_like(diagnostics.tGrid)
        self.c_helicity  = np.zeros_like(diagnostics.tGrid)
        self.m_helicity  = np.zeros_like(diagnostics.tGrid)
        self.psi_l2      = np.zeros_like(diagnostics.tGrid)
        self.circulation = np.zeros_like(diagnostics.tGrid)
        
        
        self.x = np.zeros(diagnostics.nx+1)
        self.y = np.zeros(diagnostics.ny+1)
        
        self.xpc = np.zeros(diagnostics.nx+2)
        self.ypc = np.zeros(diagnostics.ny+2)
        
        self.x[0:-1] = self.diagnostics.xGrid
        self.x[  -1] = self.x[-2] + self.diagnostics.hx
        
        self.y[0:-1] = self.diagnostics.yGrid
        self.y[  -1] = self.y[-2] + self.diagnostics.hy
        
        self.xpc[0:-1] = self.x
        self.xpc[  -1] = self.xpc[-2] + self.diagnostics.hx
        self.xpc[:] -= 0.5 * self.diagnostics.hx
        
        self.ypc[0:-1] = self.y
        self.ypc[  -1] = self.ypc[-2] + self.diagnostics.hy
        self.ypc[:] -= 0.5 * self.diagnostics.hy
        
        self.A       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.J       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.P       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.O       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
        self.Bx      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.By      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.Vx      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.Vy      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
        # set up figure/window size
        self.figure = plt.figure(num=None, figsize=(16,9))
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.2, wspace=0.25)
        plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.05)
        
        # set up plot title
        self.title = self.figure.text(0.5, 0.97, 't = %1.6f' % (diagnostics.tGrid[self.iTime]), horizontalalignment='center')
        
        # set up tick formatter
        majorFormatter = ScalarFormatter(useOffset=False)
        ## -> limit to 1.1f precision
        majorFormatter.set_powerlimits((-1,+1))
        majorFormatter.set_scientific(True)

        # add data for zero timepoint
        self.add_timepoint()
        
        # set up plots
        self.axes  = {}
        self.conts = {}
        self.cbars = {}
        self.lines = {}
        self.vecs  = {}
        self.pcms  = {}
        
        self.update_boundaries()
        
        
        # create subplots
        gs = gridspec.GridSpec(4, 6)
        self.gs = gs
        
        self.axes["Bx"] = plt.subplot(gs[0,0])
        self.axes["By"] = plt.subplot(gs[1,0])
        self.axes["Vx"] = plt.subplot(gs[2,0])
        self.axes["Vy"] = plt.subplot(gs[3,0])
        self.axes["A"]  = plt.subplot(gs[0:2,1:3])
        self.axes["J"]  = plt.subplot(gs[0:2,3:5])
        self.axes["P"]  = plt.subplot(gs[2:4,1:3])
        self.axes["O"]  = plt.subplot(gs[2:4,3:5])
        self.axes["E"]  = plt.subplot(gs[0,5])
        self.axes["C"]  = plt.subplot(gs[1,5])
        self.axes["H"]  = plt.subplot(gs[2,5])
        self.axes["L"]  = plt.subplot(gs[3,5])
        
        self.axes["Bx"].set_xlim((self.x[0], self.x[-1]))
        self.axes["Bx"].set_ylim((self.y[0], self.y[-1])) 
        self.axes["By"].set_xlim((self.x[0], self.x[-1]))
        self.axes["By"].set_ylim((self.y[0], self.y[-1])) 
        self.axes["Vx"].set_xlim((self.x[0], self.x[-1]))
        self.axes["Vx"].set_ylim((self.y[0], self.y[-1])) 
        self.axes["Vy"].set_xlim((self.x[0], self.x[-1]))
        self.axes["Vy"].set_ylim((self.y[0], self.y[-1])) 
        
        self.axes["Bx"].set_title('$B_{x} (x,y)$')
        self.axes["By"].set_title('$B_{y} (x,y)$')
        self.axes["Vx"].set_title('$V_{x} (x,y)$')
        self.axes["Vy"].set_title('$V_{y} (x,y)$')
        
        self.pcms["Bx"] = self.axes["Bx"].pcolormesh(self.xpc, self.ypc, self.Bx.T, norm=self.Bnorm, cmap=plt.get_cmap('viridis'))
        self.pcms["By"] = self.axes["By"].pcolormesh(self.xpc, self.ypc, self.By.T, norm=self.Bnorm, cmap=plt.get_cmap('viridis'))
        self.pcms["Vx"] = self.axes["Vx"].pcolormesh(self.xpc, self.ypc, self.Vx.T, norm=self.Vnorm, cmap=plt.get_cmap('viridis'))
        self.pcms["Vy"] = self.axes["Vy"].pcolormesh(self.xpc, self.ypc, self.Vy.T, norm=self.Vnorm, cmap=plt.get_cmap('viridis'))
        
        self.axes["A"].set_xlim((self.x[0], self.x[-1]))
        self.axes["A"].set_ylim((self.y[0], self.y[-1])) 
        self.axes["J"].set_xlim((self.x[0], self.x[-1]))
        self.axes["J"].set_ylim((self.y[0], self.y[-1])) 
        self.axes["P"].set_xlim((self.x[0], self.x[-1]))
        self.axes["P"].set_ylim((self.y[0], self.y[-1])) 
        self.axes["O"].set_xlim((self.x[0], self.x[-1]))
        self.axes["O"].set_ylim((self.y[0], self.y[-1]))
        
        self.axes["A"].set_title('$A (x,y)$')
        self.axes["J"].set_title('$J (x,y)$')
        self.axes["P"].set_title('$\Phi (x,y)$')
        self.axes["O"].set_title('$\Omega (x,y)$')
        
        self.pcms["A"] = self.axes["A"].pcolormesh(self.xpc, self.ypc, self.A.T, norm=self.Anorm, cmap=plt.get_cmap('viridis'))
        self.pcms["J"] = self.axes["J"].pcolormesh(self.xpc, self.ypc, self.J.T, norm=self.Jnorm, cmap=plt.get_cmap('viridis'))
        self.pcms["P"] = self.axes["P"].pcolormesh(self.xpc, self.ypc, self.P.T, norm=self.Pnorm, cmap=plt.get_cmap('viridis'))
        self.pcms["O"] = self.axes["O"].pcolormesh(self.xpc, self.ypc, self.O.T, norm=self.Onorm, cmap=plt.get_cmap('viridis'))
        
        
        tStart, tEnd, xStart, xEnd = self.get_timerange()

        self.lines["E"], = self.axes["E"].plot(self.diagnostics.tGrid[tStart:tEnd], self.energy    [tStart:tEnd])
        self.lines["C"], = self.axes["C"].plot(self.diagnostics.tGrid[tStart:tEnd], self.c_helicity[tStart:tEnd])
        self.lines["H"], = self.axes["H"].plot(self.diagnostics.tGrid[tStart:tEnd], self.m_helicity[tStart:tEnd])
        self.lines["L"], = self.axes["L"].plot(self.diagnostics.tGrid[tStart:tEnd], self.psi_l2    [tStart:tEnd])
        
        if self.diagnostics.plot_energy:
            self.axes["E"].set_title('$E (t)$')
        else:
            self.axes["E"].set_title('$\Delta E (t)$')

        if self.diagnostics.plot_c_helicity:
            self.axes["C"].set_title('$C_{\mathrm{CH}} (t)$')
        else:
            self.axes["C"].set_title('$\Delta C_{\mathrm{CH}} (t)$')
        
        if self.diagnostics.plot_m_helicity:
            self.axes["H"].set_title('$C_{\mathrm{MH}} (t)$')
        else:
            self.axes["H"].set_title('$\Delta C_{\mathrm{MH}} (t)$')
        
        if self.diagnostics.plot_psi_l2:
            self.axes["L"].set_title('$C_{L^2_\psi} (t)$')
        else:
            self.axes["L"].set_title('$\Delta C_{L^2_\psi} (t)$')
        
        self.axes["E"].set_xlim((xStart,xEnd)) 
        self.axes["C"].set_xlim((xStart,xEnd)) 
        self.axes["H"].set_xlim((xStart,xEnd)) 
        self.axes["L"].set_xlim((xStart,xEnd)) 
        
        self.axes["E"].yaxis.set_major_formatter(majorFormatter)
        self.axes["C"].yaxis.set_major_formatter(majorFormatter)
        self.axes["H"].yaxis.set_major_formatter(majorFormatter)
        self.axes["L"].yaxis.set_major_formatter(majorFormatter)
        
        # switch off some ticks
        plt.setp(self.axes["Bx"].get_xticklabels(), visible=False)
        plt.setp(self.axes["By"].get_xticklabels(), visible=False)
        plt.setp(self.axes["Vx"].get_xticklabels(), visible=False)
        plt.setp(self.axes["A" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["J" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["E" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["C" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["H" ].get_xticklabels(), visible=False)
        
        
        self.update()
        
    
    def compute_norm(self, data):
        data_min = +1e40
        data_max = -1e40
        
        data_min = min(data_min, data.min() )
        data_max = max(data_max, data.max() )

#         abs_max = max(abs(data_min), abs(data_max))
#         norm    = colors.Normalize(vmin=-2.*abs_max, vmax=+2.*abs_max)

        delta = 0.1 * (data_max - data_min)
        norm  = colors.Normalize(vmin=data_min-delta, vmax=data_max+delta)
        
        return norm 
    
    
    def compute_norm_vec(self, data1, data2):
        data_min = +1e40
        data_max = -1e40
        
        data_min = min(data_min, data1.min() )
        data_min = min(data_min, data2.min() )
        
        data_max = max(data_max, data1.max() )
        data_max = max(data_max, data2.max() )

        delta = 0.1 * (data_max - data_min)
        norm  = colors.Normalize(vmin=data_min-delta, vmax=data_max+delta)
        
        return norm 
    
    
    def update_boundaries(self):
        self.Bnorm = self.compute_norm_vec(self.diagnostics.Bx, self.diagnostics.By)
        self.Vnorm = self.compute_norm_vec(self.diagnostics.Vx, self.diagnostics.Vy)
        self.Anorm = self.compute_norm(self.diagnostics.A)
        self.Jnorm = self.compute_norm(self.diagnostics.J)
        self.Onorm = self.compute_norm(self.diagnostics.O)
        self.Pnorm = self.compute_norm(self.diagnostics.P)
        
    
    def update(self, final=False):
        
        if not (self.iTime == 1 or (self.iTime-1) % self.nPlot == 0 or self.iTime-1 == self.nTime):
            return
        
#        self.update_boundaries()

        for ckey, cont in self.conts.items():
            for coll in cont.collections:
                self.axes[ckey].collections.remove(coll)
        
        
        self.A [0:-1, 0:-1] = self.diagnostics.A [:,:]
        self.A [  -1, 0:-1] = self.diagnostics.A [0,:]
        self.A [   :,   -1] = self.A[:,0]
        
        self.J [0:-1, 0:-1] = self.diagnostics.J [:,:]
        self.J [  -1, 0:-1] = self.diagnostics.J [0,:]
        self.J [   :,   -1] = self.J[:,0]
        
        self.P [0:-1, 0:-1] = self.diagnostics.P [:,:]
        self.P [  -1, 0:-1] = self.diagnostics.P [0,:]
        self.P [   :,   -1] = self.P[:,0]
        
        self.O [0:-1, 0:-1] = self.diagnostics.O [:,:]
        self.O [  -1, 0:-1] = self.diagnostics.O [0,:]
        self.O [   :,   -1] = self.O[:,0]
        
        self.Bx[0:-1, 0:-1] = self.diagnostics.Bx[:,:]
        self.Bx[  -1, 0:-1] = self.diagnostics.Bx[0,:]
        self.Bx[   :,   -1] = self.Bx[:,0]
        
        self.By[0:-1, 0:-1] = self.diagnostics.By[:,:]
        self.By[  -1, 0:-1] = self.diagnostics.By[0,:]
        self.By[   :,   -1] = self.By[:,0]
        
        self.Vx[0:-1, 0:-1] = self.diagnostics.Vx[:,:]
        self.Vx[  -1, 0:-1] = self.diagnostics.Vx[0,:]
        self.Vx[   :,   -1] = self.Vx[:,0]
        
        self.Vy[0:-1, 0:-1] = self.diagnostics.Vy[:,:]
        self.Vy[  -1, 0:-1] = self.diagnostics.Vy[0,:]
        self.Vy[   :,   -1] = self.Vy[:,0]
        
        
        self.pcms["Bx"].set_array(self.Bx.T.ravel())
        self.pcms["By"].set_array(self.By.T.ravel())
        self.pcms["Vx"].set_array(self.Vx.T.ravel())
        self.pcms["Vy"].set_array(self.Vy.T.ravel())
        
        self.pcms["A" ].set_array(self.A.T.ravel())
        self.pcms["J" ].set_array(self.J.T.ravel())
        self.pcms["P" ].set_array(self.P.T.ravel())
        self.pcms["O" ].set_array(self.O.T.ravel())
        
        self.conts["A"] = self.axes["A"].contour(self.x, self.y, self.A.T, 20, colors='white')
        self.conts["P"] = self.axes["P"].contour(self.x, self.y, self.P.T, 20, colors='white')
        
        
        tStart, tEnd, xStart, xEnd = self.get_timerange()
        
        self.lines["E"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["E"].set_ydata(self.energy[tStart:tEnd])
        self.axes ["E"].relim()
        self.axes ["E"].autoscale_view()
        self.axes ["E"].set_xlim((xStart,xEnd)) 
        
        self.lines["C"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["C"].set_ydata(self.c_helicity[tStart:tEnd])
        self.axes ["C"].relim()
        self.axes ["C"].autoscale_view()
        self.axes ["C"].set_xlim((xStart,xEnd)) 
        
        self.lines["H"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["H"].set_ydata(self.m_helicity[tStart:tEnd])
        self.axes ["H"].relim()
        self.axes ["H"].autoscale_view()
        self.axes ["H"].set_xlim((xStart,xEnd)) 
        
        self.lines["L"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["L"].set_ydata(self.psi_l2[tStart:tEnd])
        self.axes ["L"].relim()
        self.axes ["L"].autoscale_view()
        self.axes ["L"].set_xlim((xStart,xEnd)) 
        
        if self.write:
            filename = self.prefix + str('%06d' % (self.iTime-1)) + '.png'
            plt.savefig(filename, dpi=100)
        else:
            plt.draw()
            plt.show(block=final)
            plt.pause(0.01)
    
    
    def add_timepoint(self):
        
        self.m_energy   [self.iTime] = self.diagnostics.m_energy
        self.k_energy   [self.iTime] = self.diagnostics.k_energy
        
        if self.diagnostics.plot_energy:
            self.energy     [self.iTime] = self.diagnostics.energy
        else:
            self.energy     [self.iTime] = self.diagnostics.energy_error

        if self.diagnostics.plot_c_helicity:
            self.c_helicity [self.iTime] = self.diagnostics.c_helicity
        else:
            self.c_helicity [self.iTime] = self.diagnostics.c_helicity_error
        
        if self.diagnostics.plot_m_helicity:
            self.m_helicity [self.iTime] = self.diagnostics.m_helicity
        else:
            self.m_helicity [self.iTime] = self.diagnostics.m_helicity_error
        
        if self.diagnostics.plot_psi_l2:
            self.psi_l2     [self.iTime] = self.diagnostics.psi_l2
        else:
            self.psi_l2     [self.iTime] = self.diagnostics.psi_l2_error
        
        if self.diagnostics.plot_circulation:
            self.circulation[self.iTime] = self.diagnostics.circulation
        else:
            self.circulation[self.iTime] = self.diagnostics.circulation_error
        
        
        self.title.set_text('t = %1.6f' % (self.diagnostics.tGrid[self.iTime]))
        
        self.iTime += 1
        
    
    def get_timerange(self):
        tStart = self.iTime - (self.nTime+1)
        tEnd   = self.iTime
        
        if tStart < 0:
            tStart = 0
        
        xStart = self.diagnostics.tGrid[tStart]
        xEnd   = self.diagnostics.tGrid[tStart+self.nTime]
        
        return tStart, tEnd, xStart, xEnd
    
