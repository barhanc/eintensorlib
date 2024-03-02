import numpy as np
from functools import reduce
from itertools import product

Data = np.ndarray

zeros = np.zeros
ones = np.ones
array = np.array
random = lambda shape: np.random.normal(0.0, 1.0, shape)
allclose = np.allclose

max = np.maximum
heaviside = np.heaviside
einsum = np.einsum


def delta(shape):
    dim = len(shape)
    out = ones(shape + shape)
    for x in product(*[range(n) for n in shape + shape]) if dim > 0 else []:
        out[x] = 1 if reduce(lambda a, b: a and b, [x[i] == x[dim + i] for i in range(dim)]) else 0
    return out
