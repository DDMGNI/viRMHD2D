'''
Created on Apr 06, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import matplotlib
#matplotlib.use('Cairo')
matplotlib.use('AGG')
#matplotlib.use('PDF')


#import StringIO
import argparse
#import os

#import numpy as np
#import h5py

from diagnostics import Diagnostics 


class movie(object):
    '''
    
    '''


    def __init__(self, hdf5_file, nPlot=1, ntMax=0, write=False):
        '''
        Constructor
        '''
        
        self.diagnostics = Diagnostics(hdf5_file)
        
        if ntMax > 0 and ntMax < self.diagnostics.nt:
            self.nt = ntMax
        else:
            self.nt = self.diagnostics.nt
        
        self.nPlot = nPlot
        self.plot  = PlotMHD2D(self.diagnostics, self.diagnostics.nt, self.nt, nPlot, write)
        
    
    def init(self):
        self.update(0)
    
    
    def update(self, itime, final=False):
        self.diagnostics.read_from_hdf5(itime)
        self.diagnostics.update_invariants(itime)
        
        if itime > 0:
            self.plot.add_timepoint()
        
        return self.plot.update(final=final)
    
    
    def run(self, write=False):
        for itime in range(1, self.nt+1):
            print("it = %4i" % (itime))
            self.update(itime, final=(itime == self.nt))
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vlasov-Poisson Solver in 1D')
    
    parser.add_argument('hdf5_file', metavar='<run.hdf5>', type=str,
                        help='Run HDF5 File')
    parser.add_argument('-np', metavar='i', type=int, default=1,
                        help='plot every i\'th frame')
    parser.add_argument('-ntmax', metavar='i', type=int, default=0,
                        help='limit to i points in time')
    parser.add_argument('-o', metavar='<run.mp4>', type=str, default=None,
                        help='output video file')    
    parser.add_argument('-fps', metavar='i', type=int, default=1,
                        help='frames per second')    
    
    args = parser.parse_args()
    
    from movieplot_J import PlotMHD2D
    
    
    print
    print("Replay run with " + args.hdf5_file)
    print
    
    pyvp = movie(args.hdf5_file, ntMax=args.ntmax, nPlot=args.np, write=True)
    
    pyvp.run()
    
    print
    print("Replay finished.")
    print
    
