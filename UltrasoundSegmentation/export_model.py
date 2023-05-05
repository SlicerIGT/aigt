from torch import jit
from torch import nn

from UNet import UNet
import torch

import argparse
import logging
import os


# Parse command line arguments

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--in_channels", type=int, default=1)
parser.add_argument("--out_channels", type=int, default=1)
parser.add_argument("--log_level", type=str, default="INFO")
parser.add_argument("--log_file", type=str)
args = parser.parse_args()


# Set up logging into file or console

if args.log_file is not None:
    log_file = os.path.join(args.output_dir, args.log_file)
    logging.basicConfig(filename=log_file, filemode="w", level=args.log_level)
    print(f"Logging to file {log_file}.")
else:
    logging.basicConfig(level=args.log_level)  # Log to console


# Load model

logging.info(f"Loading model from {args.model_path}.")
logging.info(f"Input channels: {args.in_channels}.")
logging.info(f"Output channels: {args.out_channels}.")

model = UNet(in_channels=args.in_channels, out_channels=args.out_channels)
model.load_state_dict(torch.load(args.model_path))

# Export model to TorchScript using tracing

model = model.to("cpu")
model.eval()
example_input = torch.rand(1, args.in_channels, 128, 128)
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save(os.path.join(args.output_dir, "model_traced.pt"))
logging.info(f"Traced model saved to {os.path.join(args.output_dir, 'model_traced.pt')}.")
