import numpy as np
from functools import reduce
from itertools import product

Data = np.ndarray

zeros = np.zeros
ones = np.ones
array = np.array
random = lambda shape: np.random.normal(0.0, 1.0, shape)
allclose = np.allclose

heaviside = np.heaviside
max = np.maximum
einsum = np.einsum


# TODO: This implementation is too slow. Delta shouldn't be realized unless in seed. It should be a
# function N^k -> {0,1} used in einsum (How to make einsum which supports functions <- custom cuda
# kernel?)
def delta(shape):
    dim = len(shape)
    out = ones(shape + shape)
    for x in product(*[range(n) for n in shape + shape]) if dim > 0 else []:
        out[x] = 1 if reduce(lambda a, b: a and b, [x[i] == x[dim + i] for i in range(dim)]) else 0
    return out
