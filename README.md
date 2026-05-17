# Digital Image Processing — Assignments

Four assignments from the **Digital Image Processing (DZO)** course implemented from scratch in Python using NumPy.

Each assignment covers a core topic in image processing — from intensity transforms
to frequency-domain analysis, spatial filtering, and gradient-domain compositing.

---

## Tech Stack

`Python` `NumPy` `Matplotlib` `Numba` `Jupyter Notebook`

---

## Assignments

### 1 · Pointwise Intensity Transforms ([`assignment_1_pointwise/`](assignment_1_pointwise/))

Pixel-level image transformations using Look-Up Tables (LUTs).

**Implemented from scratch:**
- `get_LUT(t_name, *args)` — generates LUTs for:
  - **Negative**: inverts pixel intensities
  - **Gamma correction**: non-linear brightness adjustment (`255 * (i/255)^γ`)
  - **Quantization**: reduces number of intensity levels to `qN`
- `compute_histogram_and_cdf(im)` — computes pixel intensity histogram and cumulative distribution function (CDF) without using `np.histogram`
- `histogram_matching(im, im_target)` — maps the CDF of a source image to match a target image's CDF, enabling automatic tone transfer

**Key concept:** All transforms operate via LUT lookup (`LUT[im]`) — O(1) per pixel regardless of transform complexity.

---

### 2 · Fourier Transform & Frequency Domain ([`assignment_2_fourier/`](assignment_2_fourier/))

Synthesis and analysis of images in the frequency domain.

**Implemented from scratch:**
- `image_generator(signal_type, image_size, **params)` — generates test signals:
  - `constant` — flat field
  - `harmonic` — 2D cosine wave with frequency `(u, v)` and phase `φ`
  - `square` / `circle` — binary geometric shapes
  - `gaussian` — 2D Gaussian with sigma `σ`
  - `gabor` — Gabor filter (Gaussian-modulated harmonic) — key primitive in texture analysis and CNNs
- `cosine_envelope(image_size)` — 2D Hanning window to suppress spectral leakage before FFT

**Key concept:** Understanding how spatial structures (edges, textures, periodic patterns) manifest in the magnitude spectrum — foundational for CNNs and signal processing.

---

### 3 · Spatial Filtering & Convolution ([`assignment_3_filtering/`](assignment_3_filtering/))

Manual convolution and kernel design, optimized with Numba JIT compilation.

**Implemented from scratch:**
- `convolve2d(im, kernel)` — full 2D convolution with **replicate border padding**, implemented as a pure Python nested loop
- `convolve2d_faster` — same function compiled with **Numba `@njit`** for significant speedup
- `get_kernel(kernel_type, half_size, **params)` — generates spatial kernels:
  - `average` — box blur
  - `G` — isotropic Gaussian blur
  - `Gx` / `Gy` — Gaussian first derivatives (edge detection)
  - `L` — Laplacian of Gaussian (LoG), mean-subtracted for zero-sum
- `convolve2d_ft(im, kernel)` — convolution via **FFT** (zero-pads kernel, applies FFT convolution theorem)
- `get_kernel_sep(kernel_type, half_size, **params)` — **separable** 1D kernels (reduces O(K²·MN) to O(K·MN))
- `wiener_filtration(im, kernel, lam)` — **Wiener deconvolution** in the frequency domain: `W = conj(K) / (|K|² + λ)`
- `bilateral_filter_slow(im, kernel_s, sigma_b)` — **bilateral filter** preserving edges while smoothing noise; compiled with Numba

---

### 4 · Poisson Image Editing ([`assignment_4_poisson/`](assignment_4_poisson/))

Seamless image compositing by solving the Poisson equation in the gradient domain.

**Implemented from scratch:**
- `calc_grad(im)` — discrete forward-difference gradient `(gx, gy)` across all color channels
- `calc_div(gx, gy)` — discrete divergence (adjoint of gradient operator)
- `get_mask(gx_A, gy_A, gx_B, gy_B)` — per-pixel mask based on gradient magnitude
- `merge_grads(gx_A, gy_A, gx_B, gy_B, mask)` — blends two gradient fields
- `merge_images(im_A, im_B, mask)` — direct pixel-level alpha compositing
- `solve_FT(divergence)` — **fast Poisson solver via FFT** using eigenvalues of the discrete Laplacian
- `solve_GS(divergence, im_init, mask, num_iter)` — **Gauss-Seidel iterative solver** compiled with Numba `@njit`

**Applied to:** seamless compositing between Mona Lisa and Ginevra de' Benci, and HDR-like exposure blending.

---

## How to Run

```bash
pip install numpy matplotlib numba
jupyter notebook
```

Open any `assignment_N_*/notebook.ipynb` to explore the results interactively.

---
