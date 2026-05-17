import numpy as np

def image_generator(signal_type, image_size, **params):
    rows, cols = image_size
    
    y_coor = np.arange(rows)
    x_coor = np.arange(cols)
    y, x = np.meshgrid(y_coor, x_coor, indexing='ij')

    center_y, center_y = (rows - 1) / 2, (cols - 1) / 2
    y_dist, x_dist = y - center_y, x - center_y

    if signal_type == 'constant': 
        return np.full(image_size, params['value'], dtype=float)

    elif signal_type == 'harmonic': 
        u, v, phi = params['u'], params['v'], params['phi']
        return np.cos(2 * np.pi * (u * x / cols + v * y / rows) + phi)

    elif signal_type == 'square': 
        hs = params['half_side']
        im = np.zeros(image_size, dtype=float)
        im[(np.abs(x_dist) <= hs) & (np.abs(y_dist) <= hs)] = 1.0
        return im

    elif signal_type == 'circle': 
        r = params['radius']
        im = np.zeros(image_size, dtype=float)
        im[x_dist**2 + y_dist**2 <= r**2] = 1.0
        return im

    elif signal_type == 'gaussian': 
        s = params['sigma']
        return np.exp(-(x_dist**2 + y_dist**2) / (2 * s**2))

    elif signal_type == 'gabor': 
        u, v, phi, s = params['u'], params['v'], params['phi'], params['sigma']
        h = np.cos(2 * np.pi * (u * x / cols + v * y / rows) + phi)
        g = np.exp(-(x_dist**2 + y_dist**2) / (2 * s**2))
        return h * g

    return np.zeros(image_size)


def cosine_envelope(image_size):
    height, width = image_size
    
    window_h = np.hanning(height)
    window_w = np.hanning(width)
    
    envelope = np.outer(window_h, window_w)
    
    return envelope


# def measure_dominant_orientation(im):
#     phi = np.linspace(0, np.pi, 360)
#     magnitude = np.abs(np.fft.fftshift(np.fft.fft2(im)))

#     rows, cols = im.shape
#     center_y, center_x = rows // 2, cols // 2
    
#     y, x = np.indices((rows, cols))
#     y_dist, x_dist = y - center_y, x - center_x
    
#     measure = np.zeros_like(phi)

#     for i, angle in enumerate(phi):
#         cos, sin = np.cos(angle), np.sin(angle)
        
#         proj = x_dist*cos + y_dist* sin
#         perp_dist = np.abs(-x_dist * sin + y_dist * cos)
        
#         mask = (perp_dist < 0.7) & (proj >= 0)
#         measure[i] = np.sum(magnitude[mask])

#     return phi, measure
    

################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################

import matplotlib.pyplot as plt
import numpy as np 
import warnings 


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


    
