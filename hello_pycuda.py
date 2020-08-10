import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

from pycuda.compiler import SourceModule

BLOCKSIZE = 32
GPU_NITER = 100

MAT_SIZE_X = 1000
MAT_SIZE_Y = 1000

def gflops(sec, mat_size_x, mat_size_y):
    operations = mat_size_x * mat_size_y
    gflops = operations * 1e-9 / sec
    return gflops

block = (BLOCKSIZE, BLOCKSIZE, 1)
grid = ((MAT_SIZE_X + block[0] - 1) // block[0], (MAT_SIZE_Y + block[1] - 1) // block[1])
print("Grid = ({0}, {1}), Block = ({2}, {3})".format(grid[0], grid[1], block[0], block[1]))

start = cuda.Event()
end = cuda.Event()

h_a = numpy.random.randn(MAT_SIZE_X, MAT_SIZE_Y).astype(numpy.float32)
h_b = numpy.random.randn(MAT_SIZE_X, MAT_SIZE_Y).astype(numpy.float32)
h_d = numpy.empty_like(h_a)

d_a = gpuarray.to_gpu(h_a)
d_b = gpuarray.to_gpu(h_b)

start.record()
for i in range(GPU_NITER):
    d_d = (d_a + d_b)
    h_d = d_d.get()
end.record()
end.synchronize()

elapsed_sec = start.time_till(end) * 1e-3 / GPU_NITER

for y in range(MAT_SIZE_Y):
    for x in range(MAT_SIZE_X):
        i = y * MAT_SIZE_X + x
        if i < 10:
            print("A[%d]=%8.4f, B[%d]=%8.4f, D[%d]=%8.4f" % (i, h_a[x][y], i, h_b[x][y], i, h_d[x][y]))
        else:
            break

print("GPU: Time elapsed %f sec (%lf GFLOPS)" % (elapsed_sec, gflops(elapsed_sec, MAT_SIZE_X, MAT_SIZE_Y)))

cuda.Context.synchronize()
