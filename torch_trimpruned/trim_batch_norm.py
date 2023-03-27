import torch

def trim_batch_norm_input(batch_norm, input_mask):
    batch_norm.num_features = int(torch.count_nonzero(input_mask, dim=-1))

    if batch_norm.affine:
        batch_norm.weight = torch.nn.Parameter(batch_norm.weight[input_mask])
        batch_norm.bias = torch.nn.Parameter(batch_norm.bias[input_mask])

    if batch_norm.track_running_stats:
        batch_norm.running_mean.data = batch_norm.running_mean[input_mask]
        batch_norm.running_var.data = batch_norm.running_var[input_mask]
