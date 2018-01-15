'''
Created on Jul 02, 2012

@author: mkraus
'''

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter


class PlotMHD2D(object):
    '''
    classdocs
    '''

    def __init__(self, diagnostics, nTime=0, nPlot=1, output=0):
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
        
        if output <= 0:
            self.output = False
        else:
            self.output = True
 
        self.prefix = "viRMHD2D_potential"

        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        
        self.k_energy  = np.zeros_like(diagnostics.tGrid)
        self.m_energy  = np.zeros_like(diagnostics.tGrid)
        
        self.energy      = np.zeros_like(diagnostics.tGrid)
        self.helicity    = np.zeros_like(diagnostics.tGrid)
        
        
        self.x = np.zeros(diagnostics.nx+1)
        self.y = np.zeros(diagnostics.ny+1)
        
        self.x[0:-1] = self.diagnostics.xGrid
        self.x[  -1] = self.x[-2] + self.diagnostics.hx
        
        self.y[0:-1] = self.diagnostics.yGrid
        self.y[  -1] = self.y[-2] + self.diagnostics.hy
        
        self.A       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
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
        
        
        self.read_data()
        self.update_boundaries()
        
        
        # create main figure
        self.figure1 = plt.figure(num=1, figsize=(16,10))
        
        # create subplots
        gs = gridspec.GridSpec(4, 6)
        self.gs = gs
        
        self.axes["Bx"]    = plt.subplot(gs[0,0])
        self.axes["By"]    = plt.subplot(gs[1,0])
        self.axes["Vx"]    = plt.subplot(gs[2,0])
        self.axes["Vy"]    = plt.subplot(gs[3,0])
        self.axes["A"]     = plt.subplot(gs[0:4,1:5])
        self.axes["Emag"]  = plt.subplot(gs[0,5])
        self.axes["Evel"]  = plt.subplot(gs[1,5])
        self.axes["E"]     = plt.subplot(gs[2,5])
        self.axes["H"]     = plt.subplot(gs[3,5])
        
        
        self.pcms["Bx"] = self.axes["Bx"].pcolormesh(self.x, self.y, self.Bx.T, norm=self.Bnorm)
        self.pcms["By"] = self.axes["By"].pcolormesh(self.x, self.y, self.By.T, norm=self.Bnorm)
        self.pcms["Vx"] = self.axes["Vx"].pcolormesh(self.x, self.y, self.Vx.T, norm=self.Vnorm)
        self.pcms["Vy"] = self.axes["Vy"].pcolormesh(self.x, self.y, self.Vy.T, norm=self.Vnorm)
        
        self.axes["Bx"].set_title('$B_{x} (x,y)$')
        self.axes["By"].set_title('$B_{y} (x,y)$')
        self.axes["Vx"].set_title('$V_{x} (x,y)$')
        self.axes["Vy"].set_title('$V_{y} (x,y)$')
        
        self.axes["Bx"].set_xlim((self.x[0], self.x[-1]))
        self.axes["Bx"].set_ylim((self.y[0], self.y[-1])) 
        self.axes["By"].set_xlim((self.x[0], self.x[-1]))
        self.axes["By"].set_ylim((self.y[0], self.y[-1])) 
        self.axes["Vx"].set_xlim((self.x[0], self.x[-1]))
        self.axes["Vx"].set_ylim((self.y[0], self.y[-1])) 
        self.axes["Vy"].set_xlim((self.x[0], self.x[-1]))
        self.axes["Vy"].set_ylim((self.y[0], self.y[-1])) 
        
        self.axes["A"].set_title('$\psi_e (x,y)$')
        
        self.axes["A"].set_xlim((self.x[0], self.x[-1]))
        self.axes["A"].set_ylim((self.y[0], self.y[-1]))
        
         
        tStart, tEnd, xStart, xEnd = self.get_timerange()

        self.lines["Emag" ], = self.axes["Emag" ].plot(self.diagnostics.tGrid[tStart:tEnd], self.m_energy [tStart:tEnd])
        self.lines["Evel" ], = self.axes["Evel" ].plot(self.diagnostics.tGrid[tStart:tEnd], self.k_energy [tStart:tEnd])
        self.lines["E"    ], = self.axes["E"    ].plot(self.diagnostics.tGrid[tStart:tEnd], self.energy     [tStart:tEnd])
        self.lines["H"    ], = self.axes["H"    ].plot(self.diagnostics.tGrid[tStart:tEnd], self.helicity   [tStart:tEnd])
        
        self.axes["Emag"].set_title('$E_{B} (t)$')
        self.axes["Evel"].set_title('$E_{V} (t)$')
        self.axes["E"].set_title('$\Delta E (t)$')
        self.axes["H"].set_title('$\Delta H (t)$')
        
        self.axes["Emag" ].set_xlim((xStart,xEnd)) 
        self.axes["Evel" ].set_xlim((xStart,xEnd)) 
        self.axes["E"    ].set_xlim((xStart,xEnd)) 
        self.axes["H"    ].set_xlim((xStart,xEnd)) 
        
        self.axes["Emag" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["Evel" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["E"    ].yaxis.set_major_formatter(majorFormatter)
        self.axes["H"    ].yaxis.set_major_formatter(majorFormatter)
        
        
        # switch off some ticks
        plt.setp(self.axes["Bx"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["By"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Vx"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Emag" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Evel" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["E"    ].get_xticklabels(), visible=False)
        
        
        # create psi figure
        self.figure2 = plt.figure(num=2, figsize=(10,10))
#         plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.4)
        self.axes_A  = plt.subplot(1,1,1)
        self.conts_A = None
        
        
        
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
    
    
    def read_data(self):
        self.A [0:-1, 0:-1] = self.diagnostics.A [:,:]
        self.A [  -1, 0:-1] = self.diagnostics.A [0,:]
        self.A [   :,   -1] = self.A[:,0]
        
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
        
        
    
    def update(self, final=False):
        
        if not (self.iTime == 1 or (self.iTime-1) % self.nPlot == 0 or self.iTime-1 == self.nTime):
            return
        
        self.read_data()
#        self.update_boundaries()

#         self.pcms["A" ].set_array(self.X.T.copy(order='F').ravel())
#         self.pcms["Bx"].set_array(self.Bx.T.ravel())
#         self.pcms["By"].set_array(self.By.T.ravel())
#         self.pcms["Vx"].set_array(self.Vx.T.ravel())
#         self.pcms["Vy"].set_array(self.Vy.T.ravel())
        
        self.axes["A" ].cla()
        self.axes["Bx"].cla()
        self.axes["By"].cla()
        self.axes["Vx"].cla()
        self.axes["Vy"].cla()

        self.conts["A" ] = self.axes_A.contour(self.x, self.y, self.A.T, 20, colors='k')
        self.axes["Bx"].pcolormesh(self.x, self.y, self.Bx.T, norm=self.Bnorm)
        self.axes["By"].pcolormesh(self.x, self.y, self.By.T, norm=self.Bnorm)
        self.axes["Vx"].pcolormesh(self.x, self.y, self.Vx.T, norm=self.Vnorm)
        self.axes["Vy"].pcolormesh(self.x, self.y, self.Vy.T, norm=self.Vnorm)

        self.axes["A" ].set_xlim((self.x[0], self.x[-1]))
        self.axes["A" ].set_ylim((self.y[0], self.y[-1]))
        self.axes["Bx"].set_xlim((self.x[0], self.x[-1]))
        self.axes["Bx"].set_ylim((self.y[0], self.y[-1]))
        self.axes["By"].set_xlim((self.x[0], self.x[-1]))
        self.axes["By"].set_ylim((self.y[0], self.y[-1]))
        self.axes["Vx"].set_xlim((self.x[0], self.x[-1]))
        self.axes["Vx"].set_ylim((self.y[0], self.y[-1]))
        self.axes["Vy"].set_xlim((self.x[0], self.x[-1]))
        self.axes["Vy"].set_ylim((self.y[0], self.y[-1]))
        
        
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
        
        self.lines["H"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["H"].set_ydata(self.helicity[tStart:tEnd])
        self.axes ["H"].relim()
        self.axes ["H"].autoscale_view()
        self.axes ["H"].set_xlim((xStart,xEnd)) 
        
        
#         if self.conts_A is not None:
#             for coll in self.conts_A.collections:
#                 self.conts_A.collections.remove(coll)
#                 
        self.axes_A.cla()
        self.conts_A = self.axes_A.contour(self.x, self.y, self.A.T, 20, colors='k')
        self.axes_A.set_xlim((self.x[0], self.x[-1]))
        self.axes_A.set_ylim((self.y[0], self.y[-1]))    
        
        
        for i in range(2):
            plt.figure(i+1)
            plt.draw()
         

        if self.output:
            plt.figure(1)
            filename = self.prefix + str('_%06d' % (self.iTime-1)) + '.png'
            plt.savefig(filename, dpi=100)
            
            plt.figure(2)
            filename = self.prefix + str('_A_%06d' % (self.iTime-1)) + '.pdf'
            plt.savefig(filename)

    
    
    def add_timepoint(self):
        
        self.m_energy [self.iTime] = self.diagnostics.m_energy
        self.k_energy [self.iTime] = self.diagnostics.k_energy
        self.energy   [self.iTime] = self.diagnostics.energy_error
        self.helicity [self.iTime] = self.diagnostics.c_helicity_error
        
        self.title.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        
        self.iTime += 1
        
    
    def get_timerange(self):
        tStart = self.iTime - (self.nTime+1)
        tEnd   = self.iTime
        
        if tStart < 1:
            tStart = 1
        
        if tEnd < 1:
            tEnd = 1
        
        xStart = self.diagnostics.tGrid[tStart]
        xEnd   = self.diagnostics.tGrid[tStart+self.nTime-1]
        
        return tStart, tEnd, xStart, xEnd
    
