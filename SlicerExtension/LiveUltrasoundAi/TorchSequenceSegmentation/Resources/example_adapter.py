"""
Template adapter script for TorchSequenceSegmentation.

Copy this file into the same folder as your model's weights, rename it to
something ending in "_adapter.py" (e.g. "my_model_adapter.py"), and edit
load_model() / predict() below to match your model. The module will
discover it automatically alongside any TorchScript ".pt" models in the
folder you select as the model directory.

Only two functions are required. Everything about how the model is built
and how its output is turned into a single-channel prediction is up to you,
so this works for any architecture, any framework, and any custom pre/post
processing.
"""

import os
import torch
import numpy as np


# Optional: tell the UI what square input size this model expects.
# If omitted, the user sets "Model input size" manually in the GUI.
INPUT_SIZE = 128


def load_model(model_dir, device):
    """
    Called once when the model is selected.

    model_dir: absolute path to the folder containing this adapter script
               and your weight file(s).
    device:    torch.device to use ("cuda:0" or "cpu").

    Build/load your model here however you need to (torch.load with a
    state_dict, an ONNX Runtime session, a scikit-learn pickle, etc.) and
    return whatever object predict() below expects as its "model" argument.
    """
    weights_path = os.path.join(model_dir, "weights.pth")

    # --- Example: a plain PyTorch nn.Module with a saved state_dict ---
    # from my_model_definition import MyModel
    # model = MyModel()
    # model.load_state_dict(torch.load(weights_path, map_location=device))
    # model.to(device)
    # model.eval()
    # return model

    raise NotImplementedError("Fill in load_model() for your model.")


def predict(model, input_array, device):
    """
    Called once per frame.

    model:       the object returned by load_model().
    input_array: float32 numpy array, shape (C, H, W), where C = 1 + the
                 number of previous frames requested in the GUI (usually
                 C = 1 unless you're using the "previous frames" buffer).
    device:      torch.device to use.

    Must return a single-channel float numpy array of shape (H, W)
    representing the foreground prediction (e.g. a probability map, values
    roughly in [0, 1]). The rest of the pipeline (flipping, log transform,
    scan conversion, thresholding) is handled outside this function, so you
    only need to worry about getting from raw input pixels to a prediction
    map here.
    """
    # --- Example for a standard PyTorch classifier-style segmentation net ---
    # tensor = torch.from_numpy(input_array).unsqueeze(0).float().to(device)
    # with torch.inference_mode():
    #     output = model(tensor)
    # output = torch.nn.functional.softmax(output, dim=1)
    # output_array = output[0, 1, :, :].detach().cpu().numpy()
    # return output_array

    raise NotImplementedError("Fill in predict() for your model.")