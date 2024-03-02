import cupy
import math
import numpy as np
from time import perf_counter

# size of the vectors
size = 2**20
# allocating and populating the vectors
print(size)

print("Allocating memory . . .")
time_s = perf_counter()
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)

a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
time_e = perf_counter()
print(f"Done {1_000 *(time_e -time_s):.3f}ms")

vector_add_cuda_code = r"""
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   if ( item < size )
   {
      C[item] = A[item] + B[item];
   }
}
"""
vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")

threads_per_block = 1024
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

time_s = perf_counter()
vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size))
time_e = perf_counter()
print(f"CUDA: {1000*(time_e-time_s):3.3f}ms")

time_s = perf_counter()
c_cpu = a_cpu + b_cpu
time_e = perf_counter()
print(f"CPU : {1000*(time_e - time_s):3.3f}ms")

# if np.allclose(a_cpu + b_cpu, c_gpu):
#  print("Correct results!")
# else:
#  print("Wrong results!")
