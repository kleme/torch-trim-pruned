import unittest
import torch
from trim_pt_linear_layer import trim_linear_layer

class TestTrimLinearLayer(unittest.TestCase):
    def setUp(self):
        self.layer = torch.nn.Linear(5, 6)

    def test_output_mask_with_no_rows_removed(self):
        out_mask = trim_linear_layer(self.layer)
        self.assertEqual(out_mask.tolist(), [True, True, True, True, True, True])

    def test_output_mask_with_rows_removed(self):
        self.layer.weight.data[2] = 0.0
        self.layer.weight.data[5] = 0.0
        out_mask = trim_linear_layer(self.layer)
        self.assertEqual(out_mask.tolist(), [True,  True, False, True, True, False])

    def test_without_input_mask_with_output_rows_removed(self):
        self.layer.weight.data[0] = 0.0
        self.layer.weight.data[2] = 0.0
        self.layer.weight.data[5] = 0.0
        trim_linear_layer(self.layer)
        self.assertEqual(self.layer.in_features, 5)
        self.assertEqual(self.layer.weight.shape[1], 5)
        self.assertEqual(self.layer.out_features, 3)
        self.assertEqual(self.layer.weight.shape[0], 3)
        self.assertEqual(self.layer.bias.shape[0], 3)

    def test_with_input_mask_without_output_rows_removed_(self):
        input_mask = torch.ones(self.layer.in_features, dtype=torch.bool)
        input_mask[3] = False
        input_mask[4] = False
        out_mask = trim_linear_layer(self.layer, input_mask)
        self.assertEqual(out_mask.tolist(), [True, True, True, True, True, True])
        self.assertEqual(self.layer.in_features, 3)
        self.assertEqual(self.layer.weight.shape[1], 3)
        self.assertEqual(self.layer.out_features, 6)
        self.assertEqual(self.layer.weight.shape[0], 6)
        self.assertEqual(self.layer.bias.shape[0], 6)
