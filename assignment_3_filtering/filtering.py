import numpy as np
from numba import njit
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def convolve2d(im, kernel): 
    M, N = im.shape
    K, L = kernel.shape
    kh, lh = K // 2, L // 2
    
    result = np.zeros_like(im)

    for i in range(M):
        for j in range(N):
            acc = 0.0
            for ki in range(K):
                for kj in range(L):
                    ii = i + (ki - kh)
                    jj = j + (kj - lh)
                    
                    if ii < 0: ii = 0
                    elif ii >= M: ii = M - 1
                    
                    if jj < 0: jj = 0
                    elif jj >= N: jj = N - 1
                    
                    acc += im[ii, jj] * kernel[K - 1 - ki, L - 1 - kj]
            
            result[i, j] = acc

    return result


# Numba-compiled version of the function - DO NOT REMOVE :}
convolve2d_faster = njit(cache=True)(convolve2d)

def get_kernel(kernel_type, half_size, **params):
    sz = 2*half_size + 1 
    kernel = np.zeros((sz, sz), dtype=float)
    y, x = np.ogrid[-half_size:half_size+1, -half_size:half_size+1]

    if kernel_type == 'average': 
        kernel[:] = 1.0 / (sz * sz)
        return kernel 
    
    if kernel_type not in ['G', 'Gx', 'Gy', 'L']: 
        raise ValueError("Unknown kernel type.")
    
    sigma = params.get('sigma')
    if sigma is None or len(params) != 1: 
        raise ValueError(f"{kernel_type} requires: sigma")

    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    if kernel_type == 'G': 
        kernel = g / np.sum(g)
    elif kernel_type == 'Gx': 
        kernel = g * (-x / sigma**2)
    elif kernel_type == 'Gy': 
        kernel = g * (-y / sigma**2)
    elif kernel_type == 'L': 
        kernel = g * ((x**2 + y**2 - 2 * sigma**2) / sigma**4)
        kernel -= np.mean(kernel) 

    return kernel


def convolve2d_ft(im, kernel): 
    M, N = im.shape

    K_padded = np.zeros((M, N))
    kh, lh = kernel.shape[0] // 2, kernel.shape[1] // 2
    K_padded[:kernel.shape[0], :kernel.shape[1]] = kernel
    
    K_padded = np.roll(K_padded, (-kh, -lh), axis=(0, 1))
    
    im_ft = fft2(im)
    kernel_ft = fft2(K_padded)
    
    result = np.real(ifft2(im_ft * kernel_ft))
    return result


def get_kernel_sep(kernel_type, half_size, **params):
    sz = 2*half_size + 1 
    kernel_x = np.zeros((1, sz), dtype=float)
    kernel_y = np.zeros((sz, 1), dtype=float)
    coords = np.arange(-half_size, half_size+1)

    if kernel_type == 'average': 
        kernel_x[:] = 1.0 / sz
        kernel_y[:] = 1.0 / sz
        return kernel_x, kernel_y 
    
    sigma = params.get('sigma')
    g = np.exp(-coords**2 / (2 * sigma**2))
    g_norm = g / np.sum(g)

    if kernel_type == 'G': 
        kernel_x[0, :] = g_norm
        kernel_y[:, 0] = g_norm
    elif kernel_type == 'Gx': 
        dg = g * (-coords / sigma**2)
        kernel_x[0, :] = dg
        kernel_y[:, 0] = g_norm
    elif kernel_type == 'Gy': 
        dg = g * (-coords / sigma**2)
        kernel_x[0, :] = g_norm
        kernel_y[:, 0] = dg

    return kernel_x, kernel_y

def wiener_filtration(im, kernel, lam): 
    M, N = im.shape
    kh, lh = kernel.shape[0] // 2, kernel.shape[1] // 2
    
    K_padded = np.zeros((M, N))
    K_padded[:kernel.shape[0], :kernel.shape[1]] = kernel
    K_padded = np.roll(K_padded, (-kh, -lh), axis=(0, 1))
    
    I_ft = fft2(im)
    K_ft = fft2(K_padded)
    
    K_ft_conj = np.conj(K_ft)
    W = K_ft_conj / (np.abs(K_ft)**2 + lam)
    
    result = np.real(ifft2(I_ft * W))
    return result


def bilateral_filter_slow(im, kernel_s, sigma_b): 
    M, N = im.shape
    K, L = kernel_s.shape
    kh, lh = K // 2, L // 2
    
    result = np.zeros_like(im)
    
    for i in range(M):
        for j in range(N):
            center_val = im[i, j]
            w_sum = 0.0
            pixel_sum = 0.0
            
            for ki in range(K):
                for kj in range(L):
                    ii = i + (ki - kh)
                    jj = j + (kj - lh)
                    
                    if ii < 0: ii = 0
                    elif ii >= M: ii = M - 1
                    
                    if jj < 0: jj = 0
                    elif jj >= N: jj = N - 1
                    
                    current_val = im[ii, jj]

                    intensity_dist_sq = (current_val - center_val)**2
                    kernel_range = np.exp(-intensity_dist_sq / (2 * sigma_b**2))
                    
                    w = kernel_s[ki, kj] * kernel_range
                    
                    pixel_sum += current_val * w
                    w_sum += w
            
            if w_sum != 0:
                result[i, j] = pixel_sum / w_sum
            else:
                result[i, j] = center_val
                
    return result

# compiled version
bilateral_filter = njit(cache=True)(bilateral_filter_slow)


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################


import matplotlib.pyplot as plt
import numpy as np 
import warnings 

def show_tuple(ims, captions = ['image', 'kernel', 'conv result'], cmap = 'gray', colorbars = True): 
    N = len(ims)
    plt.figure(figsize = (1+3*N,3), constrained_layout = True)
    for i, (im, caption) in enumerate( zip(ims, captions) ): 
        plt.subplot(1, N, i+1); plt.imshow(im, cmap = cmap); plt.title(caption); 
        if colorbars: 
            plt.colorbar() 


def show_FT_pair(im, additive_c = 1e-1, 
                 cmap_image = 'gray', cmap_spectrum = 'viridis', 
                 figsize = (8, 2.5), fig_title='image and FT pair'): 
    '''
    Displays an image and magnitude of its Fourier transform (fftshift-ed), next to each other.
    When displaying the magnitude spectrum, a the following transformation is used: 
    imshow( log(additive_c + abs(fftshift(fft2(im)))) )
:    '''
    
    fig = plt.figure(figsize = figsize, constrained_layout = True)
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap = cmap_image)

    plt.subplot(1, 2, 2)
    plt.imshow(np.fft.fftshift( np.log(0.1 + abs( np.fft.fft2( im ) )) ) , cmap = cmap_spectrum)
    fig.suptitle(fig_title)

def show_FT_triplet(im, additive_c = 1e-1, 
                 cmap_image = 'gray', cmap_spectrum = 'jet', 
                 figsize = (8, 3), show_frequency_labels = True): 
    '''
    Displays an image and magnitude of its Fourier transform, and the same magnitude fftshift-ed. 
    When displaying the magnitude spectrum, a the following transformation is used: 
    imshow( log(additive_c + abs(fftshift(fft2(im)))) )
    show_frequency_labels: see code.
    '''
    
    def f2f(f, N, centered):
        c = N//2

        if not centered:
            f = f - N if f>c else f 
        else: 
            f -= c
        return f 


    def display_frequency_labels(image_size, centered = True):
        M, N = image_size
        
        if M>7 or N>7 or not show_frequency_labels: # only place labels if size of image is low. 
            return 
        
        for v in range(M): 
            for u in range(N): 
                u_displ, v_displ = f2f(u, N, centered), f2f(v, M, centered)
                if (u_displ, v_displ) == (0, 0): 
                    text = 'DC'
                else:
                    text = str( (u_displ, v_displ)) 

                plt.text(u, v, text, horizontalalignment='center', 
                         verticalalignment='center', 
                         #backgroundcolor='black', 
                         color = 'white',
                         fontsize = 'xx-small')
        
        # show also dividing lines between pixels 
        for v in range(M-1):
            plt.plot([v+.5, v+.5], [-0.5, N-.5], color='w', lw = 0.5)
        for u in range(N-1):
            plt.plot([-0.5, M-.5], [u+.5, u+.5], color='w', lw = 0.5)
                


    plt.figure(figsize = figsize, constrained_layout = True)
    plt.subplot(1, 3, 1)
    plt.imshow(im, cmap = cmap_image)
    plt.title('image')

    plt.subplot(1, 3, 2)
    X = np.fft.fft2(im)
    plt.imshow( np.log( additive_c + np.abs(X) ), cmap = cmap_spectrum )
    display_frequency_labels(im.shape, centered = False)
    plt.title('mag. spectrum')

    plt.subplot(1, 3, 3)
    X = np.fft.fftshift(X)
    plt.imshow( np.log( additive_c + np.abs(X) ), cmap = cmap_spectrum )
    display_frequency_labels(im.shape, centered = True)
    plt.title('mag. spectrum, fft-shifted')

    print(f'image: min={im.min():1.1e}, max={im.max():1.1e}.')


    
