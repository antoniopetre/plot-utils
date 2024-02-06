from numba import njit, prange, config
#import tbb

#config.THREADING_LAYER = 'tbb'

@njit
def fillVector(branch):
    l = []
    for value in branch:
        l.append(value)
    return l

@njit
def fillVectorOfVectors(branch):
    l = []
    for value in branch:
        i = []
        for v in value:
            i.append(v)
        l.append(i)
    return l
