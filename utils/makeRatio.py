import numpy as np
from numba import njit

@njit
def makeRatio(numValues, denomValues):
    ratio,error = [], []
    for i,j in zip(numValues, denomValues):
        if j!=0 and i!= 0:
            ratio.append(i/j)
            error.append(i/j * np.sqrt(1./i + 1./j))
        else:
            ratio.append(np.nan)
            error.append(np.nan)
    return ratio, error
