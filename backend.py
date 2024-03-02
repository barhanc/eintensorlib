import cupy as cp
from itertools import product
from functools import reduce

Data_t = cp.ndarray

set_printoptions = cp.set_printoptions
max = cp.maximum
zeros = cp.zeros
ones = cp.ones
array = cp.array
random = cp.random.rand
normal = lambda shape: cp.random.normal(size=shape)
allclose = cp.allclose
heaviside = cp.heaviside
einsum = cp.einsum
float32 = cp.float32


def delta(shape):
    dim = len(shape)
    out = cp.ones(shape + shape)
    for x in product(*[range(n) for n in shape + shape]) if dim > 0 else []:
        out[x] = 1 if reduce(lambda a, b: a and b, [x[i] == x[dim + i] for i in range(dim)]) else 0
    return out
