import importlib.util
import pathlib
import unittest
import numpy as np

ROOT = pathlib.Path(__file__).parents[1]

def load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class FilterTests(unittest.TestCase):
    def test_padding_and_convolution(self):
        f = load("hw2/filters.py")
        image = np.zeros((5, 5)); image[2, 2] = 1
        kernel = np.ones((3, 3))
        self.assertEqual(f.zero_pad(image, 1, 2).shape, (7, 9))
        np.testing.assert_allclose(f.conv_nested(image, kernel), f.conv_fast(image, kernel))
        np.testing.assert_allclose(f.conv_fast(image, kernel), f.conv_faster(image, kernel))

    def test_normalized_match_is_bounded(self):
        f = load("hw2/filters.py")
        result = f.normalized_cross_correlation(np.arange(25.).reshape(5, 5), np.ones((3, 3)))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(((result >= -1) & (result <= 1)).all())

class EdgeTests(unittest.TestCase):
    def test_gaussian_and_hough(self):
        e = load("hw3/edge.py")
        kernel = e.gaussian_kernel(5, 1.4)
        self.assertAlmostEqual(float(kernel.sum()), 1.0)
        edges = np.zeros((10, 10), dtype=bool); edges[:, 5] = True
        accumulator, rhos, thetas = e.hough_transform(edges)
        self.assertEqual(accumulator.shape, (len(rhos), len(thetas)))
        self.assertGreater(int(accumulator.max()), 0)

    def test_canny_returns_binary_edges(self):
        e = load("hw3/edge.py")
        image = np.zeros((24, 24)); image[8:16, 8:16] = 255
        result = e.canny(image, kernel_size=5, sigma=1.0, high=30, low=10)
        self.assertEqual(result.dtype, np.bool_)
        self.assertGreater(int(result.sum()), 0)

if __name__ == "__main__":
    unittest.main()
