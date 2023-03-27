import unittest
import torch
import sys

sys.path.append("../")
from trim_batch_norm import trim_batch_norm_input

class TestBatchNorm(unittest.TestCase):
    def setUp(self):
        self.input_size = 4
        self.input_mask = torch.ones(self.input_size, dtype=torch.bool)
        self.input_mask[0] = False
        self.input_mask[3] = False

    def test_trim_batch_norm_input_affine_track_running_stats(self):
        batch_norm = torch.nn.BatchNorm1d(self.input_size)
        trim_batch_norm_input(batch_norm, self.input_mask)
        self.assertEqual(batch_norm.num_features, 2)
        self.assertEqual(batch_norm.weight.shape[0], 2)
        self.assertEqual(batch_norm.bias.shape[0], 2)
        self.assertEqual(batch_norm.running_mean.shape[0], 2)
        self.assertEqual(batch_norm.running_var.shape[0], 2)

    def test_trim_batch_norm_input_affine_without_track_running_stats(self):
        batch_norm = torch.nn.BatchNorm1d(self.input_size, track_running_stats=False)
        trim_batch_norm_input(batch_norm, self.input_mask)
        self.assertEqual(batch_norm.num_features, 2)
        self.assertEqual(batch_norm.weight.shape[0], 2)
        self.assertEqual(batch_norm.bias.shape[0], 2)

    def test_trim_batch_norm_input_without_affine_and_track_running_stats(self):
        batch_norm = torch.nn.BatchNorm1d(self.input_size, affine=False, track_running_stats=False)
        trim_batch_norm_input(batch_norm, self.input_mask)
        self.assertEqual(batch_norm.num_features, 2)
