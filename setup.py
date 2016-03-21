#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from distutils.core      import setup
from distutils.extension import Extension
from Cython.Distutils    import build_ext
from Cython.Build        import cythonize
from Cython.Compiler.Options import directive_defaults

import os
from os.path import join, isdir

# os.environ["CC"]  = "gcc-mp-5"
# os.environ["CXX"] = "g++-mp-5"

# os.environ["CC"]  = "clang"
# os.environ["CXX"] = "clang++"


INCLUDE_DIRS = []
LIBRARY_DIRS = []
LIBRARIES    = []
CARGS        = ['-O3', '-march=native', '-std=c99', '-Wno-unused-function', '-Wno-unneeded-internal-declaration']
#CARGS       += ['-Wa,-q'] # needed by some versions of gcc on MacOSX when march is set to native
LARGS        = []
MACROS       = []
# MACROS       = [('CYTHON_TRACE', '1')]

# profiling
# directive_defaults['profile']   = True
# directive_defaults['linetrace'] = True
# directive_defaults['binding']   = True


# PETSc
PETSC_DIR  = os.environ['PETSC_DIR']
PETSC_ARCH = os.environ.get('PETSC_ARCH', '')

if PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH)):
    INCLUDE_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'include'),
                     join(PETSC_DIR, 'include')]
    LIBRARY_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'lib')]
else:
    if PETSC_ARCH: pass # XXX should warn ...
    INCLUDE_DIRS += [join(PETSC_DIR, 'include')]
    LIBRARY_DIRS += [join(PETSC_DIR, 'lib')]

LIBRARIES    += ['petsc']

# NumPy
import numpy
INCLUDE_DIRS += [numpy.get_include()]

# PETSc for Python
import petsc4py
INCLUDE_DIRS += [petsc4py.get_include()]

# MPI
IMPI_DIR = '/afs/@cell/common/soft/intel/ics2013/impi/4.1.3/intel64'

if isdir(IMPI_DIR):
    INCLUDE_DIRS += [join(IMPI_DIR, 'include')]
    LIBRARY_DIRS += [join(IMPI_DIR, 'lib')]

if isdir('/opt/local/include/openmpi-gcc5'):
    INCLUDE_DIRS += ['/opt/local/include/openmpi-gcc5']
if isdir('/opt/local/lib/openmpi-gcc5'):
    LIBRARY_DIRS += ['/opt/local/lib/openmpi-gcc5']


extension_list = ["PETScDerivatives",
                  "PETScNonlinearSolverArakawaJ1CFD2",
                  "PETScNonlinearSolverArakawaJ1CFD2DOF2",
                  "PETScPreconditionerArakawaJ1CFD2",
                  "PETScPreconditionerArakawaJ1CFD2Vec",
                  "PETScOhmsLawArakawaJ1",
                  "PETScPoissonCFD2",
                  "PETScVorticityArakawaJ1"]


ext_modules = [
        Extension(ext,
                  sources=[ext + ".pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS,
                  define_macros=MACROS
                 ) for ext in extension_list]
                
setup(
    name = 'PETSc RMHD Solver',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
