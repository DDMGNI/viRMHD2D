#!/bin/bash
#
#$ -cwd
#
#$ -l h_rt=24:00:00
#
#$ -P  tokp
#$ -pe impi_hydra 16
#
#$ -o /tokp/scratch/mkraus/viRMHD2D/current_sheet_512x256.$JOB_ID.out
#$ -e /tokp/scratch/mkraus/viRMHD2D/current_sheet_512x256.$JOB_ID.err
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N viRMHD2D
#

RUNID=current_sheet_512x256


module load intel/14.0
module load mkl/11.1
module load impi/4.1.3
module load hdf5-mpi/1.8.12

module load py33-python
module load py33-cython
module load py33-numpy
module load py33-scipy
module load py33-configobj
module load py33-mpi4py
module load py33-petsc4py


export LD_PRELOAD=/afs/@cell/common/soft/intel/ics2013/14.0/mkl/lib/intel64/libmkl_core.so:$LD_PRELOAD
export LD_PRELOAD=/afs/@cell/common/soft/intel/ics2013/14.0/mkl/lib/intel64/libmkl_intel_thread.so:$LD_PRELOAD
export LD_PRELOAD=/afs/@cell/common/soft/intel/ics2013/14.0/compiler/lib/intel64/libiomp5.so:$LD_PRELOAD


export RUN_DIR=/u/mkraus/Codes/viRMHD2D
cd $RUN_DIR

mpiexec -perhost 16 -l -n 16 python3.3 petsc_mhd2d.py runs/$RUNID.cfg 
