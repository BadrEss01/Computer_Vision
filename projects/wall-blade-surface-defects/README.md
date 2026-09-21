# Visual surface-defect inspection extension

This folder contains a new, reproducible computer-vision baseline proposed as a follow-up to the wind-blade inspection system described in the attached cooperative thesis.

The original report describes a wall-climbing robot using passive suction cups, a belt-driven mechanism, a swivel chassis and a force sensor, with inspection identified as a future application. This folder is **not part of the original 2022 prototype**. It is a separate extension for evaluating images captured from a future camera mounted on the robot.

## Baseline

`detect_defects.py` uses a deliberately explainable pipeline:

1. convert an image to grayscale;
2. estimate a smooth local background;
3. calculate absolute local contrast;
4. threshold candidate anomalies;
5. remove small connected components;
6. return bounding boxes and a binary mask.

This is a baseline for experimentation, not a validated turbine-blade inspection model. It can detect visual contrast anomalies, but it cannot distinguish cracks, dirt, shadows, reflections or structural damage without labelled data.

## Run

```bash
python -m pip install numpy opencv-python
python detect_defects.py path/to/image.jpg --output annotated.png
```

The next evidence-building step is a labelled dataset of blade-surface images, followed by a train/validation/test split, precision/recall/F1, false-positive analysis and comparison against a learned detector.

## Relationship to the thesis

The thesis is cited as system context only. It documents the proposed wind-blade inspection system and prototype design; it does not report this defect detector, a camera integration, a dataset or validated defect metrics.
