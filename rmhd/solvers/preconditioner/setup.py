#!/usr/bin/env python

from rmhd.setup import *

extension_list = ["PETScPreconditionerArakawaJ1CFD2",
                  "PETScPreconditionerArakawaJ1CFD2Vec",
                  "PETScPreconditionerArakawaJ1CFD2DOF2Vec"]

make_extension(extension_list)
