"""Utilities function that should be loaded before everything else."""

import os
import sys

# https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
# https://github.com/numpy/numpy/issues/11826
def set_n_cpus(n_cpus):
    
    assert 'numpy' not in sys.modules, 'NumPy is already important, NO EFFECT!'
    
    n_cpus = str(n_cpus)
    os.environ['OMP_NUM_THREADS'] = n_cpus
    os.environ['OPENBLAS_NUM_THREADS'] = n_cpus
    os.environ['MKL_NUM_THREADS'] = n_cpus
    os.environ['VECLIB_MAXIMUM_THREADS'] = n_cpus
    os.environ['NUMEXPR_NUM_THREADS'] = n_cpus