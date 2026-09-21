import importlib.util
from pathlib import Path
import unittest
import cv2
import numpy as np

path = Path(__file__).parents[1] / "detect_defects.py"
spec = importlib.util.spec_from_file_location("detect_defects", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class SurfaceDefectTests(unittest.TestCase):
    def test_detects_contrast_region(self):
        image = np.full((120, 160, 3), 150, dtype=np.uint8)
        image[50:70, 70:90] = 20
        mask, boxes = module.detect_defects(image, threshold=15, min_area=10)
        self.assertGreater(mask.sum(), 0)
        self.assertTrue(any(w > 0 and h > 0 for _, _, w, h in boxes))

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            module.detect_defects(np.array([]))

if __name__ == "__main__":
    unittest.main()
