import cupy as cp
import numpy as np
import math

from typing import Final, Callable, Iterable
from abc import ABC, abstractmethod
from itertools import product
from functools import reduce


set_printoptions = cp.set_printoptions
max = cp.maximum
zeros = cp.zeros
ones = cp.ones
array = cp.array
random = cp.random.rand
allclose = cp.allclose
heaviside = cp.heaviside
einsum = cp.einsum
float32 = cp.float32


def delta(shape):
    dim = len(shape)
    out = np.ones(shape + shape)
    for x in product(*[range(n) for n in shape + shape]) if dim > 0 else []:
        out[x] = 1 if reduce(lambda a, b: a and b, [x[i] == x[dim + i] for i in range(dim)]) else 0
    return out
