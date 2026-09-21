"""Canny and Hough-transform building blocks for the CV coursework."""
import numpy as np


def conv(image, kernel):
    image, kernel = np.asarray(image), np.asarray(kernel)
    hi, wi = image.shape
    hk, wk = kernel.shape
    padded = np.pad(image, ((hk // 2, hk // 2), (wk // 2, wk // 2)), mode="edge")
    out = np.zeros((hi, wi), dtype=float)
    flipped = np.flip(kernel)
    for y in range(hi):
        for x in range(wi):
            out[y, x] = np.sum(padded[y:y + hk, x:x + wk] * flipped)
    return out


def gaussian_kernel(size, sigma):
    if size <= 0 or size % 2 == 0 or sigma <= 0:
        raise ValueError("size must be positive and odd; sigma must be positive")
    axis = np.arange(size) - size // 2
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def partial_x(image):
    return conv(image, np.array([[-1, 0, 1]], dtype=float))


def partial_y(image):
    return conv(image, np.array([[-1], [0], [1]], dtype=float))


def gradient(image):
    gx, gy = partial_x(image), partial_y(image)
    magnitude = np.sqrt(gx * gx + gy * gy)
    theta = np.mod(np.degrees(np.arctan2(gy, gx)), 360.0)
    return magnitude, theta


def non_maximum_suppression(G, theta):
    G, theta = np.asarray(G), np.mod(theta, 180.0)
    h, w = G.shape
    out = np.zeros_like(G)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            t = theta[y, x]
            if t < 22.5 or t >= 157.5:
                a, b = G[y, x - 1], G[y, x + 1]
            elif t < 67.5:
                a, b = G[y - 1, x + 1], G[y + 1, x - 1]
            elif t < 112.5:
                a, b = G[y - 1, x], G[y + 1, x]
            else:
                a, b = G[y - 1, x - 1], G[y + 1, x + 1]
            if G[y, x] >= a and G[y, x] >= b:
                out[y, x] = G[y, x]
    return out


def double_thresholding(img, high, low):
    if low > high:
        raise ValueError("low threshold must not exceed high threshold")
    strong = np.asarray(img) > high
    weak = (np.asarray(img) > low) & ~strong
    return strong, weak


def get_neighbors(y, x, H, W):
    return [(i, j) for i in (y - 1, y, y + 1) for j in (x - 1, x, x + 1)
            if 0 <= i < H and 0 <= j < W and (i, j) != (y, x)]


def link_edges(strong_edges, weak_edges):
    strong = np.asarray(strong_edges, dtype=bool)
    weak = np.asarray(weak_edges, dtype=bool).copy()
    edges = strong.copy()
    stack = list(zip(*np.nonzero(strong)))
    while stack:
        y, x = stack.pop()
        for ny, nx in get_neighbors(y, x, *strong.shape):
            if weak[ny, nx]:
                weak[ny, nx] = False
                edges[ny, nx] = True
                stack.append((ny, nx))
    return edges


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    smoothed = conv(img, gaussian_kernel(kernel_size, sigma))
    magnitude, theta = gradient(smoothed)
    thinned = non_maximum_suppression(magnitude, theta)
    strong, weak = double_thresholding(thinned, high, low)
    return link_edges(strong, weak)


def hough_transform(img):
    img = np.asarray(img, dtype=bool)
    h, w = img.shape
    diag_len = int(np.ceil(np.hypot(w, h)))
    rhos = np.arange(-diag_len, diag_len + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    ys, xs = np.nonzero(img)
    if len(xs):
        rho_values = xs[:, None] * np.cos(thetas) + ys[:, None] * np.sin(thetas)
        rho_indices = np.rint(rho_values).astype(int) + diag_len
        for row in rho_indices:
            np.add.at(accumulator, (row, np.arange(len(thetas))), 1)
    return accumulator, rhos, thetas
