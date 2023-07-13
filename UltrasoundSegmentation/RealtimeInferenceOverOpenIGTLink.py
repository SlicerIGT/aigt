"""
Implements an OpenIGTLink client that expect pyigtl.ImageMessage and returns pyigtl.ImageMessage with YOLOv5 inference added to the image.
Arguments:
    model: string path to the torchscript file you intend to use
    input device name: This is the device name the client is listening to
    output device name: The device name the client outputs to
    host: the server's IP the client connects to.
    input port: port used for receiving data from the PLUS server over OpenIGTLink
    output port: port used for sending data to Slicer over OpenIGTLink
    target size: target quadratic size the model resizes to internally for predictions. Does not affect the actual output size
    confidence threshold: only bounding boxes above the given threshold will be visualized.
    line thickness: line thickness of drawn bounding boxes. Also affects font size of class names and confidence
"""

import argparse
import traceback
import sys
import json
import numpy as np
import cv2
from pathlib import Path
import pyigtl
import torch


ROOT = Path(__file__).parent.resolve()

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--input-device-name", type=str, default="Image_Reference")
    parser.add_argument("--output-device-name", type=str, default="Inference")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--input-port", type=int, default=18944)
    parser.add_argument("--output-port", type=int, default=18945)
    try:
        return parser.parse_args()
    except SystemExit as err:
        traceback.print_exc()
        sys.exit(err.code)


# runs the client in an infinite loop, waiting for messages from the server. Once a message is received,
# the message is processed and the inference is sent back to the server as a pyigtl ImageMessage.
def run_client(args):
    input_client = pyigtl.OpenIGTLinkClient(host=args.host, port=args.input_port)
    output_server = pyigtl.OpenIGTLinkServer(port=args.output_port)
    model = None

    while True:
        message = input_client.wait_for_message(args.input_device_name, timeout=-1)

        if isinstance(message, pyigtl.ImageMessage):
            if model is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # Load model
                model_path = args.model if Path(args.model).is_absolute() else f'{str(ROOT)}/{args.model}'
                extra_files = {"config.json": ""}
                model = torch.jit.load(model_path, _extra_files=extra_files).to(device)
                config = json.loads(extra_files["config.json"])
                input_size = config["shape"][-1]

            # Resize image to model input size
            orig_img_size = message.image.shape
            image = preprocess_input(message.image, input_size).to(device)
        
            # Run inference
            with torch.inference_mode():
                prediction = model(image)

            if isinstance(prediction, list):
                prediction = prediction[0]
            prediction = postprocess_prediction(prediction, orig_img_size)

            image_message = pyigtl.ImageMessage(prediction, device_name=args.output_device_name)
            output_server.send_message(image_message, wait=True)
        else:
            print(f'Unexpected message format. Message:\n{message}')


def preprocess_input(image, input_size):
    image = cv2.resize(image[0, :, :], (input_size, input_size))  # default is bilinear
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    return image


def postprocess_prediction(prediction, original_size):
    prediction = prediction.squeeze().detach().cpu().numpy() * 255
    prediction = cv2.resize(prediction[1], (original_size[2], original_size[1]))
    prediction = prediction.astype(np.uint8)[np.newaxis, ...]
    return prediction


if __name__ == "__main__":
    args = parse_args()
    run_client(args)
