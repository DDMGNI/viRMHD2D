#!/usr/bin/env python

from rmhd.setup import *

extension_list = ["PETScOhmsLawArakawaJ1",
                  "PETScPoissonCFD2",
                  "PETScVorticityArakawaJ1"]

make_extension(extension_list)
