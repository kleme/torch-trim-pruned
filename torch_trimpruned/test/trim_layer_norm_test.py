import unittest
import torch
import sys

sys.path.append("../")
from trim_layer_norm import trim_layer_norm_input

class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        self.input_size = 4
        self.input_mask = torch.ones(self.input_size, dtype=torch.bool)
        self.input_mask[0] = False
        self.input_mask[3] = False

    def test_trim_layer_norm_input_with_1d_normalized_shape(self):
        layer_norm = torch.nn.LayerNorm(self.input_size, elementwise_affine=True)
        trim_layer_norm_input(layer_norm, self.input_mask)
        self.assertEqual(layer_norm.normalized_shape, (2,))
        self.assertEqual(layer_norm.weight.shape[0], 2)
        self.assertEqual(layer_norm.bias.shape[0], 2)

    def test_trim_layer_norm_input_with_1d_normalized_shape_without_elementwise_affine(self):
        layer_norm = torch.nn.LayerNorm(self.input_size, elementwise_affine=False)
        trim_layer_norm_input(layer_norm, self.input_mask)
        self.assertEqual(layer_norm.normalized_shape[0], 2)

    def test_trim_layer_norm_input_with_2d_normalized_shape(self):
        layer_norm = torch.nn.LayerNorm((3, self.input_size), elementwise_affine=True)
        trim_layer_norm_input(layer_norm, self.input_mask)
        self.assertEqual(layer_norm.normalized_shape, (3, 2))
        self.assertEqual(layer_norm.weight.shape, (3, 2))
        self.assertEqual(layer_norm.bias.shape, (3, 2))

    def test_trim_layer_norm_input_with_2d_normalized_shape_without_elementwise_affine(self):
        layer_norm = torch.nn.LayerNorm((3, self.input_size), elementwise_affine=False)
        trim_layer_norm_input(layer_norm, self.input_mask)
        self.assertEqual(layer_norm.normalized_shape, (3, 2))
