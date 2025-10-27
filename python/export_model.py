import os

import torch

def save_to_onnx(model: torch.nn.Module, ref_input: torch.Tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.onnx.export(
        model,
        ref_input,
        path,
        export_params=True,
        do_constant_folding=True,

        # Support dynamic batch size
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},  # Define dynamic batch size for input
            'output': {0: 'batch_size'}  # Define dynamic batch size for output
        }
    )