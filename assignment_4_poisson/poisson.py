
import numpy as np
from numba import njit
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# ============================================================
# GRADIENT & DIVERGENCE
# ============================================================

def calc_grad(im):
    im = im.astype(np.float64)

    gx = np.zeros_like(im)
    gy = np.zeros_like(im)

    gx[:, :-1, :] = im[:, 1:, :] - im[:, :-1, :]
    gy[:-1, :, :] = im[1:, :, :] - im[:-1, :, :]

    return gx, gy


def calc_div(gx, gy):
    divergence = np.zeros_like(gx)

    divergence[:, 0, :] += gx[:, 0, :]
    divergence[:, 1:, :] += gx[:, 1:, :] - gx[:, :-1, :]

    divergence[0, :, :] += gy[0, :, :]
    divergence[1:, :, :] += gy[1:, :, :] - gy[:-1, :, :]

    return divergence

# ============================================================
# MASK & MERGING
# ============================================================

def get_mask(gx_A, gy_A, gx_B, gy_B):
    H, W, C = gx_A.shape

    mag_A = np.zeros((H, W))
    mag_B = np.zeros((H, W))

    for c in range(C):
        mag_A += gx_A[:, :, c] ** 2 + gy_A[:, :, c] ** 2
        mag_B += gx_B[:, :, c] ** 2 + gy_B[:, :, c] ** 2

    mask = np.zeros((H, W))
    mask[mag_A <= mag_B] = 1

    return mask


def merge_grads(gx_A, gy_A, gx_B, gy_B, mask):
    gx = gx_A.copy()
    gy = gy_A.copy()

    for c in range(gx_A.shape[2]):
        gx[:, :, c] = gx_A[:, :, c] * (1 - mask) + gx_B[:, :, c] * mask
        gy[:, :, c] = gy_A[:, :, c] * (1 - mask) + gy_B[:, :, c] * mask

    return gx, gy


def merge_images(im_A, im_B, mask):
    result = im_A.copy()

    for c in range(im_A.shape[2]):
        result[:, :, c] = im_A[:, :, c] * (1 - mask) + im_B[:, :, c] * mask

    return result


# ============================================================
# POISSON SOLVERS
# ============================================================


def solve_FT(divergence):
    H, W, C = divergence.shape

    O = np.zeros_like(divergence).astype(np.float64)

    y = np.arange(H)
    x = np.arange(W)

    yy, xx = np.meshgrid(y, x, indexing='ij')

    denom = 2 * np.cos(2 * np.pi * xx / W) + 2 * np.cos(2 * np.pi * yy / H) - 4
    denom[0, 0] = 1

    for c in range(C):
        div_fft = fft2(divergence[:, :, c])
        sol_fft = div_fft / denom
        sol_fft[0, 0] = 0
        O[:, :, c] = np.real(ifft2(sol_fft))

    return O


@njit
def solve_GS(divergence, im_init, mask, num_iter=100):
    H, W, C = im_init.shape
    O = im_init.copy().astype(np.float64)

    for it in range(num_iter):
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if mask[y, x] != 0:
                    for c in range(C):
                        O[y, x, c] = (O[y, x - 1, c] + O[y, x + 1, c] + O[y - 1, x, c] + O[y + 1, x, c] -divergence[y, x, c]) / 4.0

    return O


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################


import matplotlib.pyplot as plt
import numpy as np 
import warnings 

def normalized(x): 
    return ( x - x.min() ) / (x.max() - x.min())

def signed_power(x, beta): 
    '''
    returns:
      np.sign(x)*np.power( np.abs(x), beta )
    '''
    return np.sign(x)*np.power( np.abs(x), beta )


def show_tuple(ims, captions = ['image', 'kernel', 'conv result'], cmap = 'gray', colorbars = True): 
    N = len(ims)
    plt.figure(figsize = (1+3*N,3), constrained_layout = True)
    for i, (im, caption) in enumerate( zip(ims, captions) ): 
        plt.subplot(1, N, i+1); plt.imshow(im, cmap = cmap); plt.title(caption); 
        if colorbars: 
            plt.colorbar() 
