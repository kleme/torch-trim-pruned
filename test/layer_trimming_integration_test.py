import unittest
import torch
import sys

sys.path.append("../torch_trimpruned")
from trim_linear_layer import trim_linear_layer
from trim_batch_norm import trim_batch_norm_input
from trim_layer_norm import trim_layer_norm_input
from trim_weight_norm_linear_layer import trim_weight_norm_linear_layer


class TestModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.linear_layer1 = torch.nn.utils.weight_norm(torch.nn.Linear(input_size, 6))
        self.linear_layer2 = torch.nn.Linear(6, 8)
        self.layer_norm = torch.nn.LayerNorm(8)
        self.linear_layer3 = torch.nn.Linear(8, 10)
        self.batch_norm = torch.nn.BatchNorm1d(10)
        self.linear_layer4 = torch.nn.Linear(10, output_size)

    def forward(self, x):
        result = self.relu(self.linear_layer1(x))
        result = self.relu(self.linear_layer2(result))
        result = self.layer_norm(result)
        result = self.relu(self.linear_layer3(result))
        result = self.batch_norm(result)
        return self.linear_layer4(result)

class TestLayerTrimmingIntegration(unittest.TestCase):
    def setUp(self):
        self.input_size = 3
        self.output_size = 2
        self.model = TestModel(self.input_size, self.output_size)
        self.model.eval()
        # Prune some neurons
        self.model.linear_layer1.weight_v.data[2] = 0.0
        self.model.linear_layer2.weight.data[1] = 0.0
        self.model.linear_layer2.weight.data[3] = 0.0
        self.model.linear_layer3.weight.data[0] = 0.0
        self.model.linear_layer3.weight.data[2] = 0.0
        self.model.linear_layer3.weight.data[4] = 0.0
        # Trim the layers
        input_mask = trim_weight_norm_linear_layer(self.model.linear_layer1)
        input_mask = trim_linear_layer(self.model.linear_layer2, input_mask)
        trim_layer_norm_input(self.model.layer_norm, input_mask)
        input_mask = trim_linear_layer(self.model.linear_layer3, input_mask)
        trim_batch_norm_input(self.model.batch_norm, input_mask)
        trim_linear_layer(self.model.linear_layer4, input_mask)

    def test_trimmed_model_structure(self):
        self.assertEqual(self.model.linear_layer1.weight_v.shape, (5, self.input_size))
        self.assertEqual(self.model.linear_layer2.weight.shape, (8-2, 5))
        self.assertEqual(self.model.linear_layer3.weight.shape, (10-3, 8-2))
        self.assertEqual(self.model.linear_layer4.weight.shape, (self.output_size, 7))

    def test_predict_with_single_input(self):
        output = self.model(torch.rand(1, 3))
        self.assertEqual(output.size(1), self.output_size)

    def test_predict_with_batch_input(self):
        output = self.model(torch.rand(4, 3))
        self.assertEqual(output.size(1), self.output_size)
