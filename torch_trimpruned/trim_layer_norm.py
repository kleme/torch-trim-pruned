import torch

def trim_layer_norm_input(layer_norm, input_mask):
    input_size = int(torch.count_nonzero(input_mask, dim=-1))
    if len(layer_norm.normalized_shape) > 1:
        new_normalized_shape = list(layer_norm.normalized_shape)
        new_normalized_shape[-1] = input_size
        layer_norm.normalized_shape = tuple(new_normalized_shape)
        if layer_norm.elementwise_affine:
            layer_norm.weight = torch.nn.Parameter(layer_norm.weight[:, input_mask])
            layer_norm.bias = torch.nn.Parameter(layer_norm.bias[:, input_mask])
    else:
        layer_norm.normalized_shape = (input_size,)
        if layer_norm.elementwise_affine:
            layer_norm.weight = torch.nn.Parameter(layer_norm.weight[input_mask])
            layer_norm.bias = torch.nn.Parameter(layer_norm.bias[input_mask])
