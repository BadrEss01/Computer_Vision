"""Small image-filtering utilities used by the coursework notebooks.

The functions deliberately keep the assignment's same-image-size convention:
zero padding is applied around the input and the kernel is centered.
"""
import numpy as np


def conv_nested(image, kernel):
    image = np.asarray(image)
    kernel = np.asarray(kernel)
    hi, wi = image.shape
    hk, wk = kernel.shape
    padded = zero_pad(image, hk // 2, wk // 2)
    flipped = np.flip(kernel)
    out = np.zeros((hi, wi), dtype=np.result_type(image, kernel, float))
    for y in range(hi):
        for x in range(wi):
            out[y, x] = np.sum(padded[y:y + hk, x:x + wk] * flipped)
    return out


def zero_pad(image, pad_height, pad_width):
    image = np.asarray(image)
    if pad_height < 0 or pad_width < 0:
        raise ValueError("padding must be non-negative")
    return np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)),
                  mode="constant")


def conv_fast(image, kernel):
    image = np.asarray(image)
    kernel = np.asarray(kernel)
    hi, wi = image.shape
    hk, wk = kernel.shape
    padded = zero_pad(image, hk // 2, wk // 2)
    flipped = np.flip(kernel)
    out = np.zeros((hi, wi), dtype=np.result_type(image, kernel, float))
    for y in range(hi):
        for x in range(wi):
            out[y, x] = np.sum(padded[y:y + hk, x:x + wk] * flipped)
    return out


def conv_faster(image, kernel):
    # Vectorized implementation for odd-sized kernels.
    image = np.asarray(image)
    kernel = np.asarray(kernel)
    hi, wi = image.shape
    hk, wk = kernel.shape
    if hk % 2 == 0 or wk % 2 == 0:
        return conv_fast(image, kernel)
    padded = zero_pad(image, hk // 2, wk // 2)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (hk, wk))
    return np.einsum("ijkl,kl->ij", windows, np.flip(kernel))


def cross_correlation(f, g):
    return conv_fast(f, np.asarray(g))


def zero_mean_cross_correlation(f, g):
    g = np.asarray(g)
    return cross_correlation(f, g - np.mean(g))


def normalized_cross_correlation(f, g):
    f = np.asarray(f, dtype=float)
    g = np.asarray(g, dtype=float)
    hi, wi = f.shape
    hk, wk = g.shape
    padded = zero_pad(f, hk // 2, wk // 2)
    out = np.zeros((hi, wi), dtype=float)
    g0 = g - g.mean()
    gnorm = np.linalg.norm(g0)
    for y in range(hi):
        for x in range(wi):
            patch = padded[y:y + hk, x:x + wk]
            p0 = patch - patch.mean()
            pnorm = np.linalg.norm(p0)
            out[y, x] = 0.0 if gnorm == 0 or pnorm == 0 else np.sum(p0 * g0) / (pnorm * gnorm)
    return out
