import numpy as np
from backend import Data_t, LazyData_t


def get_shape(indices: str, *Ts: Data_t | LazyData_t):
    sum_inds, free_inds = indices.split("->")
    sum_inds = sum_inds.split(",")
    assert len(sum_inds) == len(Ts), "Number of index expressions doesn't match number of operands"
    shape = [None for _ in free_inds]
    for i, free_indx in enumerate(free_inds):
        for j, T in enumerate(Ts):
            for k, sum_indx in enumerate(sum_inds[j]):
                if sum_indx == free_indx and shape[i] is None:
                    shape[i] = T.shape[k]
                elif sum_indx == free_indx:
                    assert shape[i] == T.shape[k], "Axis dimensions do not match"
    for s in shape:
        assert s is not None, "One of the axis is None"
    return tuple(shape)


if __name__ == "__main__":
    indices = "ij,ij->i"
    T1 = np.ones((2, 2))
    T2 = np.ones((2, 2))
    print(get_shape(indices, T1, T2))
