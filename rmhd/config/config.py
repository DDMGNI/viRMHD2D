'''
Created on Mar 20, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from configobj import ConfigObj
from validate  import Validator

import os.path


class Config(ConfigObj):
    '''
    Run configuration.
    '''


    def __init__(self, infile, create_default=False):
        '''
        Constructor
        '''
        
        self.runspec = 'rmhd/config/runspec.cfg'
        
        ConfigObj.__init__(self, infile=infile, configspec=self.runspec, create_empty=create_default)
        
        self.validator = Validator()
        self.valid     = self.validate(self.validator, copy=True)
        
        
        if create_default:
            # create default config file
            self.write()
            
        else:
            # compute some additional grid properties
            
            if self['grid']['x1'] != self['grid']['x2']:
                self['grid']['Lx'] = self['grid']['x2'] - self['grid']['x1']
            else:
                self['grid']['x1'] = 0.0
                self['grid']['x2'] = self['grid']['Lx']
            
            if self['grid']['y1'] != self['grid']['y2']:
                self['grid']['Ly'] = self['grid']['y2'] - self['grid']['y1']
            else:
                self['grid']['y1'] = 0.0
                self['grid']['y2'] = self['grid']['Ly']
            
            self['grid']['hx'] = self['grid']['Lx'] / self['grid']['nx']
            self['grid']['hy'] = self['grid']['Ly'] / self['grid']['ny']
        
    

if __name__ == '__main__':
    '''
    Instantiates a Config object, reads default values from runspec.cfg file
    and creates a default configuration file in run.cfg.default.
    '''
    
    filename = 'run.cfg.default'
    
    if os.path.exists(filename):
        os.remove(filename)
    
    config = Config(filename, create_default=True)

