# Computer Vision Coursework

Implementations and experiments from computer-vision coursework. The repository currently covers image manipulation, linear algebra, convolution and template matching, edge detection, Canny processing and Hough-transform voting.

## Structure

- `hw1/`: image loading/manipulation and linear-algebra helpers.
- `hw2/`: convolution, cross-correlation and normalized template matching.
- `hw3/`: Gaussian filtering, image gradients, non-maximum suppression, hysteresis and Hough voting.
- `references/`: sample input/output images used by the notebooks.

## Run

Create an environment with Python 3.10+ and install:

```bash
python -m pip install numpy pillow scikit-image jupyter
```

The modules are importable directly from their assignment directories. Example:

```bash
python -m unittest discover -s tests -v
```

The tests cover numerical invariants and small synthetic images. The notebooks contain the original coursework exploration; they are not presented as production computer-vision systems.

## Portfolio status

This is coursework, not a claim of professional deployment. The implementations are documented to make the algorithms reviewable and reproducible. No proprietary data or employer code is included.
