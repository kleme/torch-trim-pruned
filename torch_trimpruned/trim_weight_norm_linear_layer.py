import torch

def trim_weight_norm_linear_layer(weight_norm_linear_layer, input_mask=None):
    if input_mask is None:
        input_mask = torch.ones(weight_norm_linear_layer.in_features, dtype=torch.bool)
    
    # Remove weight columns that are marked as removed in the input mask
    weight_norm_linear_layer.weight_v = torch.nn.Parameter(weight_norm_linear_layer.weight_v[:, input_mask])
    weight_norm_linear_layer.in_features = weight_norm_linear_layer.weight_v.shape[1]

    # Create a boolean mask for rows in weights with non-zero values.
    # The rows with only zeros are marked False and others are marked True
    row_mask = (torch.count_nonzero(weight_norm_linear_layer.weight_v, dim=-1) > 0)

    # Remove all-zero weight rows
    weight_norm_linear_layer.weight_v = torch.nn.Parameter(weight_norm_linear_layer.weight_v[row_mask])
    weight_norm_linear_layer.weight_g = torch.nn.Parameter(weight_norm_linear_layer.weight_g[row_mask])
    weight_norm_linear_layer.out_features = (row_mask == True).nonzero().shape[0]

    if weight_norm_linear_layer.bias is not None:
        # Remove bias terms for rows with all-zero weights
        weight_norm_linear_layer.bias = torch.nn.Parameter(torch.masked_select(weight_norm_linear_layer.bias, row_mask))

    # Return row mask that can be used for the next layer's input mask
    return row_mask
