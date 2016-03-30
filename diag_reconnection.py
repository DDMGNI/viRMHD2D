'''
Created on Apr 06, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

#import StringIO
import argparse
# import numpy as np
import h5py

import matplotlib
matplotlib.use('AGG')

from diagnostics import Diagnostics 
from plot_reconnection import PlotMHD2D


class replay(object):
    '''
    
    '''

    def __init__(self, hdf5_file, nPlot=1, output=0):
        '''
        Constructor
        '''
        
        self.diagnostics = Diagnostics(hdf5_file)
        
        self.nPlot = nPlot
        self.plot  = PlotMHD2D(self.diagnostics, self.diagnostics.nt, nPlot, output)
        
    
    def init(self):
        self.update(0)
    
    
    def update(self, itime):
        print(itime)
        self.diagnostics.read_from_hdf5(itime)
        self.diagnostics.update_invariants(itime)
        
        if itime > 0:
            self.plot.add_timepoint()
        
        self.plot.update()
    
    
    def run(self):
        for itime in range(1, self.diagnostics.nt+1):
            self.update(itime)
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ideal MHD Solver in 2D')
    
    parser.add_argument('hdf5_file', metavar='<run.hdf5>', type=str,
                        help='Run HDF5 File')
    parser.add_argument('-np', metavar='i', type=int, default=1,
                        help='plot every i\'th frame')    
    parser.add_argument('-o', metavar='0', type=int, default=0,
                        help='output video file')    
    
    args = parser.parse_args()
    
    print
    print("Replay run with " + args.hdf5_file)
    print
    
    pyvp = replay(args.hdf5_file, args.np, args.o)
    pyvp.run()
    
    print
    print("Replay finished.")
    print
    
