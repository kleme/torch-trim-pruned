import torch

def trim_linear_layer(linear_layer, input_mask=None):
    if input_mask is None:
        input_mask = torch.ones(linear_layer.in_features, dtype=torch.bool)
    
    # Remove weight columns that are marked as removed in the input mask
    linear_layer.weight = torch.nn.Parameter(linear_layer.weight[:, input_mask])
    linear_layer.in_features = linear_layer.weight.shape[1]

    # Create a boolean mask for rows in weights with non-zero values.
    # The rows with only zeros are marked False and others are marked True
    row_mask = (torch.count_nonzero(linear_layer.weight, dim=-1) > 0)

    # Remove all-zero weight rows
    linear_layer.weight = torch.nn.Parameter(linear_layer.weight[row_mask])
    linear_layer.out_features = torch.nonzero(row_mask == True).shape[0]

    if linear_layer.bias is not None:
        # Remove bias terms for rows with all-zero weights
        linear_layer.bias = torch.nn.Parameter(torch.masked_select(linear_layer.bias, row_mask))

    # Return row mask that can be used for the next layer's input mask
    return row_mask
