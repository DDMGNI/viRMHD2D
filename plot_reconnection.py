'''
Created on Jul 02, 2012

@author: mkraus
'''

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter


class PlotMHD2D(object):
    '''
    classdocs
    '''

    def __init__(self, diagnostics, nTime=0, nPlot=1):
        '''
        Constructor
        '''
        
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
        self.psi_l2      = np.zeros_like(diagnostics.tGrid)
        self.c_helicity  = np.zeros_like(diagnostics.tGrid)
        self.m_helicity  = np.zeros_like(diagnostics.tGrid)
        
        
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
        self.title = self.figure.text(0.5, 0.97, 't = 0.0' % (diagnostics.tGrid[self.iTime]), horizontalalignment='center') 
        
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
        gs = gridspec.GridSpec(8, 6)
        self.gs = gs
        
        self.axes["Bx"]    = plt.subplot(gs[0:2,0])
        self.axes["By"]    = plt.subplot(gs[2:4,0])
        self.axes["J"]     = plt.subplot(gs[4:6,0])
        self.axes["P"]     = plt.subplot(gs[6:8,0])
        self.axes["A"]     = plt.subplot(gs[0:4,1:3])
        self.axes["O"]     = plt.subplot(gs[0:4,3:5])
        self.axes["Vx"]    = plt.subplot(gs[4:8,1:3])
        self.axes["Vy"]    = plt.subplot(gs[4:8,3:5])
        self.axes["Emag"]  = plt.subplot(gs[0:2,5])
        self.axes["Evel"]  = plt.subplot(gs[2:4,5])
        self.axes["E"]     = plt.subplot(gs[4,5])
        self.axes["L"]     = plt.subplot(gs[5,5])
        self.axes["Hc"]    = plt.subplot(gs[6,5])
        self.axes["Hm"]    = plt.subplot(gs[7,5])
        
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
        
        self.pcms["Bx"] = self.axes["Bx"].pcolormesh(self.xpc, self.ypc, self.Bx.T, norm=self.Bnorm)
        self.pcms["By"] = self.axes["By"].pcolormesh(self.xpc, self.ypc, self.By.T, norm=self.Bnorm)
        self.pcms["Vx"] = self.axes["Vx"].pcolormesh(self.xpc, self.ypc, self.Vx.T, norm=self.Vnorm)
        self.pcms["Vy"] = self.axes["Vy"].pcolormesh(self.xpc, self.ypc, self.Vy.T, norm=self.Vnorm)
        
        self.axes["P"].set_xlim((self.x[0], self.x[-1]))
        self.axes["P"].set_ylim((self.y[0], self.y[-1])) 
        self.axes["O"].set_xlim((self.x[0], self.x[-1]))
        self.axes["O"].set_ylim((self.y[0], self.y[-1]))
        
        self.axes["A"].set_title('$A (x,y)$')
        self.axes["J"].set_title('$J (x,y)$')
        self.axes["P"].set_title('$\Phi (x,y)$')
        self.axes["O"].set_title('$\Omega (x,y)$')
        
        self.pcms["P"] = self.axes["P"].pcolormesh(self.xpc, self.ypc, self.P.T, norm=self.Pnorm)
        self.pcms["O"] = self.axes["O"].pcolormesh(self.xpc, self.ypc, self.O.T, norm=self.Onorm)
        
        
        tStart, tEnd, xStart, xEnd = self.get_timerange()

        self.lines["Emag" ], = self.axes["Emag" ].plot(self.diagnostics.tGrid[tStart:tEnd], self.m_energy  [tStart:tEnd])
        self.lines["Evel" ], = self.axes["Evel" ].plot(self.diagnostics.tGrid[tStart:tEnd], self.k_energy  [tStart:tEnd])
        self.lines["E"    ], = self.axes["E"    ].plot(self.diagnostics.tGrid[tStart:tEnd], self.energy    [tStart:tEnd])
        self.lines["L"    ], = self.axes["L"    ].plot(self.diagnostics.tGrid[tStart:tEnd], self.psi_l2    [tStart:tEnd])
        self.lines["Hc"   ], = self.axes["Hc"   ].plot(self.diagnostics.tGrid[tStart:tEnd], self.c_helicity[tStart:tEnd])
        self.lines["Hm"   ], = self.axes["Hm"   ].plot(self.diagnostics.tGrid[tStart:tEnd], self.m_helicity[tStart:tEnd])
        
        self.axes["Emag"].set_title('$E_{B} (t)$')
        self.axes["Evel"].set_title('$E_{V} (t)$')
        
        if self.diagnostics.energy_init < 1E-10:
            self.axes["E"].set_title('$E (t) - E (0)$')
        else:
            self.axes["E"].set_title('$\Delta E (t)$')
            
        if self.diagnostics.psi_l2_init < 1E-10:
            self.axes["L"].set_title('$L^2 (t) - L^2 (0)$')
        else:
            self.axes["L"].set_title('$\Delta L^2 (t)$')
            
        if self.diagnostics.c_helicity_init < 1E-10:
            self.axes["Hc"].set_title('$H_c (t) - H_c (0)$')
        else:
            self.axes["Hc"].set_title('$\Delta H_c (t)$')
            
        if self.diagnostics.c_helicity_init < 1E-10:
            self.axes["Hm"].set_title('$H_m (t) - H_m (0)$')
        else:
            self.axes["Hm"].set_title('$\Delta H_m (t)$')
            
        self.axes["Emag" ].set_xlim((xStart,xEnd)) 
        self.axes["Evel" ].set_xlim((xStart,xEnd)) 
        self.axes["E"    ].set_xlim((xStart,xEnd)) 
        self.axes["L"    ].set_xlim((xStart,xEnd)) 
        self.axes["Hc"   ].set_xlim((xStart,xEnd)) 
        self.axes["Hm"   ].set_xlim((xStart,xEnd)) 
        
        self.axes["Emag" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["Evel" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["E"    ].yaxis.set_major_formatter(majorFormatter)
        self.axes["L"    ].yaxis.set_major_formatter(majorFormatter)
        self.axes["Hc"   ].yaxis.set_major_formatter(majorFormatter)
        self.axes["Hm"   ].yaxis.set_major_formatter(majorFormatter)
        
        # switch off some ticks
        plt.setp(self.axes["Bx"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["By"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Vx"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["A"    ].get_xticklabels(), visible=False)
        plt.setp(self.axes["J"    ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Emag" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Evel" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["E"    ].get_xticklabels(), visible=False)
        plt.setp(self.axes["L"    ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Hc"   ].get_xticklabels(), visible=False)
        
        
        self.update()
        
    
    def update_boundaries(self):
        self.Bmin = +1e40
        self.Bmax = -1e40
        
        self.Bmin = min(self.Bmin, self.diagnostics.Bx.min() )
        self.Bmin = min(self.Bmin, self.diagnostics.By.min() )
        
        self.Bmax = max(self.Bmax, self.diagnostics.Bx.max() )
        self.Bmax = max(self.Bmax, self.diagnostics.By.max() )

        dB = 0.1 * (self.Bmax - self.Bmin)
        self.Bnorm = colors.Normalize(vmin=self.Bmin-dB, vmax=self.Bmax+dB)


        self.Vmin = +1e40
        self.Vmax = -1e40
        
        self.Vmin = min(self.Vmin, self.diagnostics.Vx.min() )
        self.Vmin = min(self.Vmin, self.diagnostics.Vy.min() )
        
        self.Vmax = max(self.Vmax, self.diagnostics.Vx.max() )
        self.Vmax = max(self.Vmax, self.diagnostics.Vy.max() )

        dV = 0.1 * (self.Vmax - self.Vmin)
        self.Vnorm = colors.Normalize(vmin=self.Vmin-dV, vmax=self.Vmax+dV)
        
        
        self.Omin = +1e40
        self.Omax = -1e40
        
        self.Omin = min(self.Omin, self.diagnostics.O.min() )
        self.Omax = max(self.Omax, self.diagnostics.O.max() )

        OA = max(abs(self.Omin), abs(self.Omax))
        self.Onorm = colors.Normalize(vmin=-2.*OA, vmax=+2.*OA)
        
        
        self.Pmin = +1e40
        self.Pmax = -1e40
        
        self.Pmin = min(self.Pmin, self.diagnostics.P.min() )
        self.Pmax = max(self.Pmax, self.diagnostics.P.max() )

        dP = 0.1 * (self.Pmax - self.Pmin)
        self.Pnorm = colors.Normalize(vmin=self.Pmin-dP, vmax=self.Pmax+dP)
        
    
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
        
        self.pcms["P" ].set_array(self.P.T.ravel())
        self.pcms["O" ].set_array(self.O.T.ravel())
        
        self.conts["A"] = self.axes["A"].contour(self.x, self.y, self.A.T, 20)
        self.conts["J"] = self.axes["J"].contourf(self.x, self.y, self.J.T, 20)
        
        self.axes ["A"].set_xlim((self.x[self.diagnostics.nx*1//4], self.x[self.diagnostics.nx*3//4])) 
        
        tStart, tEnd, xStart, xEnd = self.get_timerange()
        
        self.lines["Emag"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["Emag"].set_ydata(self.m_energy[tStart:tEnd])
        self.axes ["Emag"].relim()
        self.axes ["Emag"].autoscale_view()
        self.axes ["Emag"].set_xlim((xStart,xEnd)) 
        
        self.lines["Evel"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["Evel"].set_ydata(self.k_energy[tStart:tEnd])
        self.axes ["Evel"].relim()
        self.axes ["Evel"].autoscale_view()
        self.axes ["Evel"].set_xlim((xStart,xEnd)) 
        
        self.lines["E"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["E"].set_ydata(self.energy[tStart:tEnd])
        self.axes ["E"].relim()
        self.axes ["E"].autoscale_view()
        self.axes ["E"].set_xlim((xStart,xEnd)) 
        
        self.lines["L"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["L"].set_ydata(self.psi_l2[tStart:tEnd])
        self.axes ["L"].relim()
        self.axes ["L"].autoscale_view()
        self.axes ["L"].set_xlim((xStart,xEnd)) 
        
        self.lines["Hc"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["Hc"].set_ydata(self.c_helicity[tStart:tEnd])
        self.axes ["Hc"].relim()
        self.axes ["Hc"].autoscale_view()
        self.axes ["Hc"].set_xlim((xStart,xEnd)) 
        
        self.lines["Hm"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["Hm"].set_ydata(self.m_helicity[tStart:tEnd])
        self.axes ["Hm"].relim()
        self.axes ["Hm"].autoscale_view()
        self.axes ["Hm"].set_xlim((xStart,xEnd)) 
        
        
        plt.draw()
        plt.show(block=final)
        
        return self.figure
    
    
    def add_timepoint(self):
        
        self.m_energy  [self.iTime] = self.diagnostics.m_energy
        self.k_energy  [self.iTime] = self.diagnostics.k_energy
        self.energy    [self.iTime] = self.diagnostics.energy_error
        self.psi_l2    [self.iTime] = self.diagnostics.psi_l2_error
        self.c_helicity[self.iTime] = self.diagnostics.c_helicity_error
        self.m_helicity[self.iTime] = self.diagnostics.m_helicity_error
        
        self.title.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        
        self.iTime += 1
        
    
    def get_timerange(self):
        tStart = self.iTime - (self.nTime+1)
        tEnd   = self.iTime
        
        if tStart < 0:
            tStart = 0
        
        xStart = self.diagnostics.tGrid[tStart]
        xEnd   = self.diagnostics.tGrid[tStart+self.nTime]
        
        return tStart, tEnd, xStart, xEnd
    
