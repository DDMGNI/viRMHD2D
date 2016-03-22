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


    def __init__(self, infile, file_error=True):
        '''
        Constructor
        '''
        
        self.runspec = 'runspec.cfg'
        
        ConfigObj.__init__(self, infile=infile, configspec=self.runspec, file_error=file_error)
        
        self.validator = Validator()
        self.valid     = self.validate(self.validator, copy=True)
        
        
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
        
        
    
    def write_default_config(self):
        '''
        Reads default values from runspec file and creates a default
        configuration file in run.cfg.default.
        '''
        
        self.write()
        
    

if __name__ == '__main__':
    '''
    Instantiates a Config object and creates a default configuration file.
    '''
    
    filename = 'run.cfg.default'
    
    if os.path.exists(filename):
        os.remove(filename)
    
    config = Config(filename, file_error=False)
    config.write_default_config()

