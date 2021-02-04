from math import isclose

import numpy as np

from src.utils import compute_pvalue, cohen_d_stats, cohen_d

def test_compute_pvalue():
    
    assert isclose(1/101, compute_pvalue(0, np.arange(100)))

    assert isclose(2/101, compute_pvalue(1, np.arange(100)))

    assert isclose(101/101, compute_pvalue(101, np.arange(100)))


def test_choen_d():

    n = 100

    x = np.random.normal(0, 1, n)
    y = np.random.normal(2, 1.3, n)

    assert (cohen_d(x, y)
            - cohen_d_stats(np.mean(x), np.mean(y), np.std(x), np.std(y), n)
            <= 1e-9)  # may be slightly different due to differences in numerical precison
