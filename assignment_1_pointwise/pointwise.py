#!/usr/bin/python
# -*- coding: utf-8 -*-
 
from DZO.utils import uint8_to_float01, float01_to_uint8
import numpy as np
import matplotlib.pyplot as plt
import math
 
def get_LUT(t_name: str, *args):
    match t_name:
        case 'negative':
            LUT = 255-np.arange(256, dtype=np.uint8)
        case 'gamma':
            if len(args) != 1:
                raise TypeError("For 'gamma', the 2nd argument (gamma value) must be provided")
            gamma = args[0]
            LUT = np.arange(256, dtype=np.uint8)
            for i in range(256):
                LUT[i] = 255*((i/255)**gamma)
 
        case 'quantization':
            if len(args) != 1:
                raise TypeError("For 'quantization', the 2nd argument (number of quantization levels) must be provided")
            qN = int(args[0])
            x = np.arange(256, dtype=np.int32)
            idx = (x*qN)//256          

            LUT = ((idx* 255)//(qN - 1)).astype(np.uint8)
        case _:
            raise NotImplementedError('Transformation '+str+' unknown.')
    return LUT
 
def compute_histogram_and_cdf(im: np.typing.NDArray[np.uint8]):
    '''
    computes histogram of an image for all of its <0, 255> values.
     
    returns: histogram (a 1D numpy array of type np.uint64)
             cdf (cumulative sum of the histogram). 
    '''
 
    M, N = im.shape
    hist = np.zeros(256, dtype=np.uint64)
 
    for r in range(M):
        for c in range(N):
            val = int(im[r, c])
            hist[val] += 1
 
    cdf = np.cumsum(hist).astype(np.float64)
 
    total = hist.sum()
    if total > 0:
        cdf /= total
 
    return hist, cdf
 
def histogram_matching(im: np.typing.NDArray[np.uint8], im_target: np.typing.NDArray[np.uint8]):
    '''
    Compute LUT which will make im's CDF close to im_target's CDF. 
    '''
    hist1, cdf1 = compute_histogram_and_cdf(im)
    hist2, cdf2 = compute_histogram_and_cdf(im_target)
 
    LUT = np.zeros(256, dtype=np.uint8)
 
    for i in range(256):
        min_diff = 10
        best = 0
 
        for j in range(256):
            diff = abs(cdf1[i] - cdf2[j])
 
            if diff < min_diff:
                min_diff = diff
                best = j
 
        LUT[i] = best
 
    return LUT
     
 
################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################
 
import matplotlib.pyplot as plt
import numpy as np 
import warnings 
 
def transform(im: np.typing.NDArray[np.uint8], LUT: np.typing.NDArray[np.uint8]): 
    return LUT[im]
     
def show_transformed(im, LUT):
 
    plt.figure(figsize=(8, 3), constrained_layout = True)
    plt.subplot(1, 2, 1)
    plt.plot(LUT)
    plt.title('LUT')
 
    plt.subplot(1, 2, 2)
    plt.imshow(LUT[im], cmap='gray') 
    plt.title('transformed image')
 
def show_image_hist_cdf(im, bins = None):
 
    plt.figure(figsize=(10, 3), constrained_layout = True)
    plt.subplot(1, 3, 1)
    plt.imshow(im, cmap='gray') 
    plt.title('image')
 
    if len(im.shape)>2: 
        warnings.warn("You are computing histogram of multi-channel image, are you sure that's what you want?") 
 
    binN = None
    if bins is None: 
        binN = 256
    elif isinstance(bins, int): 
        binN = bins 
 
    if binN is None: 
        pass # bins are defined by user 
    else: 
        if im.dtype == np.uint8:
            bins = np.linspace(-0.5, 255.5, binN)
        elif np.issubdtype(im.dtype, np.floating):
            bins = np.linspace(0, 1, binN)
        else:
            raise TypeError('im is not float nor np.uint8')
 
    h, bin_edges = np.histogram(im, bins)
    cdf = h.cumsum().astype(float)
    cdf = cdf/h.sum()
 
    wdth = bin_edges[1]-bin_edges[0]
    plt.subplot(1, 3, 2)
    plt.bar(bin_edges[:-1], h, align='edge', width=wdth)
    #plt.plot(bin_edges[:-1]+wdth/2, h, 'r*')
    plt.xlabel('intensity')
    plt.ylabel('count')
    plt.title('histogram')
 
    plt.subplot(1, 3, 3)
    plt.bar(bin_edges[:-1], cdf, align='edge', width=wdth)
    #plt.plot(bin_edges[:-1]+wdth/2, cdf, 'r*')
    plt.xlabel('intensity')
    plt.ylabel('probability')
    plt.title('cdf')
 
    #return bin_edges, h, cdf 
 
 
 
 
################################################################################
#####                                                                      #####
#####             Below this line you may insert debugging code            #####
#####                                                                      #####
################################################################################
 
def main():
    # HERE IT IS POSSIBLE TO ADD YOUR TESTING OR DEBUGGING CODE
    pass
 
if __name__ == "__main__":
    main()