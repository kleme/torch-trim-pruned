import torch

def trim_linear_layer(linear_layer, input_mask=None):
    if input_mask is None:
        input_mask = torch.ones(linear_layer.in_features, dtype=torch.bool)
    
    # Create a boolean mask for rows in weights with non-zero values.
    # The rows with only zeros are marked with False in the mask and others with True
    row_mask = (linear_layer.weight.sum(dim=-1) != 0)
    # Remove bias terms for rows with all zeros in weights
    linear_layer.bias.data = torch.masked_select(linear_layer.bias, row_mask)
    # Remove all zero weight rows
    linear_layer.weight.data = linear_layer.weight[row_mask]
    linear_layer.out_features = (row_mask == True).nonzero().shape[0]
    # Remove columns that are not present in input
    linear_layer.weight.data = linear_layer.weight[:, input_mask]
    linear_layer.in_features = linear_layer.weight.shape[1]
    # Return row mask that can be used for the next layer's inout mask
    return row_mask
