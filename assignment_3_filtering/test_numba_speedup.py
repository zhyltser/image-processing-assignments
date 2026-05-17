from filtering import *
import time
import numpy as np

# This file tests the speedup of convolution due to numba JIT compiler. 

N = 64
K = 7

repeatN = 10

t0 = time.time()
s = np.zeros((N, N))
for it in range(repeatN):
    im = np.random.rand(N, N)
    ker = np.random.rand(K, K)
    r = convolve2d(im, ker)
    s += r

t1 = time.time() 
ave = (t1-t0) / repeatN
print(f'average time - loopy python: {ave}')

# + NUMBA 
convolve2d_faster(im[:3, :3], ker[:1, :1]) # dry run 

repeatN = 100 

t0 = time.time() 
s = np.zeros((N, N))
for it in range(repeatN):
    im = np.random.rand(N, N)
    ker = np.random.rand(K, K)
    r = convolve2d_faster(im, ker)
    s += r

t1 = time.time() 
ave_numba = (t1-t0) / repeatN
print(f'average time - loopy python + numba: {ave_numba}')

print(f'SPEEDUP: {ave / ave_numba: .1f}')
