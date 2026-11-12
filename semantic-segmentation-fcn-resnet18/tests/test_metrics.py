"""Minimal metric sanity tests; run: python -m unittest discover -s tests."""
import unittest
import torch
from segmentation.metrics import SegmentationMetrics

class MetricsTest(unittest.TestCase):
    def test_ignored_pixels_and_perfect_predictions(self):
        m=SegmentationMetrics(num_classes=3)
        m.update(torch.tensor([[0,1],[2,0]]),torch.tensor([[0,1],[2,255]]))
        stats=m.compute()
        self.assertEqual(stats["valid_pixels"],3)
        self.assertAlmostEqual(stats["pixel_accuracy"],1.0)
        self.assertAlmostEqual(stats["mean_iou"],1.0)

    def test_all_void(self):
        m=SegmentationMetrics(num_classes=3)
        m.update(torch.tensor([[0]]),torch.tensor([[255]]))
        self.assertIsNone(m.compute()["pixel_accuracy"])

if __name__=="__main__":unittest.main()
