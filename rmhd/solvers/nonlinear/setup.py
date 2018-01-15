#!/usr/bin/env python

from rmhd.setup import *

extension_list = ["PETScNonlinearSolverArakawaJ1CFD2",
                  "PETScNonlinearSolverArakawaJ1CFD2DB",
                  "PETScNonlinearSolverArakawaJ1CFD2DOF2"]

make_extension(extension_list)
