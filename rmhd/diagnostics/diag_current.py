'''
Created on Apr 06, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse

import numpy as np

import matplotlib
#matplotlib.use('Cairo')
matplotlib.use('AGG')
#matplotlib.use('PDF')

import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter


from diagnostics import Diagnostics 


class PlotMHD2D(object):
    '''
    classdocs
    '''

    def __init__(self, diagnostics, filename, ntMax=0, nPlot=1, write=False):
        '''
        Constructor
        '''
        
#        matplotlib.rc('text', usetex=True)
        matplotlib.rc('font', family='sans-serif', size='28')
        
        self.prefix = filename
        
        self.ntMax = diagnostics.nt
        
        if self.ntMax > ntMax and ntMax > 0:
            self.ntMax = ntMax
        
        self.nPlot = nPlot
        self.iTime = -1
        
        self.diagnostics = diagnostics
        
        
        self.x = np.zeros(diagnostics.nx+1)
        self.y = np.zeros(diagnostics.ny+1)
        
        self.x[0:-1] = self.diagnostics.xGrid
        self.x[  -1] = self.x[-2] + self.diagnostics.hx
        
        self.y[0:-1] = self.diagnostics.yGrid
        self.y[  -1] = self.y[-2] + self.diagnostics.hy
        
        self.J       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
        
        # set up figure/window size
        self.figure = plt.figure(num=None, figsize=(10,10))
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.25, wspace=0.2)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        
        # set up plot title
        self.title = self.figure.text(0.5, 0.95, 't = 0.0' % (diagnostics.tGrid[self.iTime]), horizontalalignment='center', fontsize=30) 
        
        # set up tick formatter
        majorFormatter = ScalarFormatter(useOffset=False)
        ## -> limit to 1.1f precision
        majorFormatter.set_powerlimits((-1,+1))
        majorFormatter.set_scientific(True)

        # create axes
        self.axes = plt.subplot(1,1,1)
        
        # add data for zero timepoint and compute boundaries
        self.add_timepoint()
        self.update_boundaries()
        
        # create contour plot
        self.conts = self.axes.contourf(self.x, self.y, self.J.T, 51, norm=self.Jnorm)
#         self.axes.pcolormesh(self.x, self.y, self.J.T, cmap=plt.get_cmap('viridis'))
#         self.axes.set_xlim((self.x[0], self.x[-1]))
#         self.axes.set_ylim((self.y[0], self.y[-1]))
        
        for tick in self.axes.xaxis.get_major_ticks():
            tick.set_pad(12)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.set_pad(8)
        
        
        # plot
        self.update()
        
    
    def read_data(self):
        
        self.J[0:-1, 0:-1] = self.diagnostics.J[:,:]
        self.J[  -1, 0:-1] = self.diagnostics.J[0,:]
        self.J[   :,   -1] = self.J[:,0]
        
    
    
    def update_boundaries(self):
        
        Jmin = min(self.diagnostics.J.min(), -self.diagnostics.J.max())
        Jmax = min(self.diagnostics.J.max(), -self.diagnostics.J.min())
        Jdiff = (Jmax - Jmin)
        
        if Jmin == Jmax:
            Jmin -= 1.
            Jmax += 1.
        
        self.Jnorm = colors.Normalize(vmin=Jmin - 0.2*Jdiff, vmax=Jmax + 0.2*Jdiff)
        self.JTicks = np.linspace(Jmin - 0.2*Jdiff, Jmax + 0.2*Jdiff, 51, endpoint=True)
        
    
    
    def update(self):
        
        if not (self.iTime == 0 or (self.iTime) % self.nPlot == 0 or self.iTime == self.ntMax):
            return
        
        self.read_data()

        for coll in self.conts.collections:
            self.axes.collections.remove(coll)
         
        self.conts = self.axes.contourf(self.x, self.y, self.J.T, 51, norm=self.Jnorm, cmap=plt.get_cmap('viridis'))
        
#         self.axes.cla()
#         self.axes.pcolormesh(self.x, self.y, self.J.T, cmap=plt.get_cmap('viridis'))
#         self.axes.set_xlim((self.x[0], self.x[-1]))
#         self.axes.set_ylim((self.y[0], self.y[-1]))
        
        plt.draw()
        
        filename = self.prefix + str('_J_%06d' % self.iTime) + '.png'
        plt.savefig(filename, dpi=300)
#         filename = self.prefix + str('_J_%06d' % self.iTime) + '.pdf'
#         plt.savefig(filename)
    
    
    def add_timepoint(self):
        self.iTime += 1
        self.title.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        
    


class Plot(object):
    '''
    
    '''


    def __init__(self, hdf5_file, nPlot=1, ntMax=0):
        '''
        Constructor
        '''
        
        self.diagnostics = Diagnostics(hdf5_file)
        
        if ntMax > 0 and ntMax < self.diagnostics.nt:
            self.nt = ntMax
        else:
            self.nt = self.diagnostics.nt
        
        self.plot = PlotMHD2D(self.diagnostics, args.hdf5_file.replace(".hdf5", ""), self.nt, nPlot)
        
    
    def update(self, itime):
        self.diagnostics.read_from_hdf5(itime)
        self.diagnostics.update_invariants(itime)
        
        if itime > 0:
            self.plot.add_timepoint()
        
        self.plot.update()
    
    
    def run(self):
        for itime in range(1, self.nt+1):
            print("it = %4i" % (itime))
            self.update(itime)
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vlasov-Poisson Solver in 1D')
    
    parser.add_argument('hdf5_file', metavar='<run.hdf5>', type=str,
                        help='Run HDF5 File')
    parser.add_argument('-np', metavar='i', type=int, default=1,
                        help='plot every i\'th frame')
    parser.add_argument('-ntmax', metavar='i', type=int, default=0,
                        help='limit to i points in time')
    
    args = parser.parse_args()
    
    
    print
    print("Replay run with " + args.hdf5_file)
    print
    
    pyvp = Plot(args.hdf5_file, ntMax=args.ntmax, nPlot=args.np)
    pyvp.run()
    
    print
    print("Replay finished.")
    print
    
