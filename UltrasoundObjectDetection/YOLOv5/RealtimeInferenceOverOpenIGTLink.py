"""
Implements an OpenIGTLink client that expect pyigtl.ImageMessage and returns pyigtl.ImageMessage with YOLOv5 inference added to the image.
Arguments:
    weights: string path to the .pt weights file used for the model
    input device name: This is the device name the client is listening to
    output device name: The device name the client outputs to
    host: the server's IP the client connects to.
    port: port used for bidirectional communication between server and client.
    target size: target quadratic size the model resizes to internally for predictions. Does not affect the actual output size
    confidence threshold: only bounding boxes above the given threshold will be visualized.
    line thickness: line thickness of drawn bounding boxes. Also affects font size of class names and confidence
"""

import argparse
import traceback
import sys
import numpy as np
import pyigtl
import numpy as np
from YOLOv5 import YOLOv5
import torch
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/lung_us_pretrained.pt")
    parser.add_argument("--input-device-name", type=str, default="Image_Reference")
    parser.add_argument("--output-device-name", type=str, default="Inference")
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18945)
    try:
        return parser.parse_args()
    except SystemExit as err:
        traceback.print_exc()
        sys.exit(err.code)


# runs the client in an infinite loop, waiting for messages from the server. Once a message is received,
# the message is processed and the inference is sent back to the server as a pyigtl ImageMessage.
def run_client(args):
    client = pyigtl.OpenIGTLinkClient(host=args.host, port=args.port)
    model = None

    while True:
        message = client.wait_for_message(args.input_device_name, timeout=-1)

        if isinstance(message, pyigtl.ImageMessage):
            if model is None:
                input_size = message.image.shape[1:3]
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                weights = args.weights if Path(args.weights).is_absolute() else f'{str(ROOT)}/{args.weights}'
                model = YOLOv5(weights=weights,
                               device=device,
                               line_thickness=args.line_thickness,
                               input_size=input_size,
                               target_size=args.target_size)

            image = preprocess_epiphan_image(message.image)

            prediction = model.predict(image, args.confidence_threshold)
            image_message = pyigtl.ImageMessage(np.flip(np.flip(prediction, axis=1), axis=2), device_name=args.output_device_name)
            client.send_message(image_message, wait=True)
        else:
            print(f'Unexpected message format. Message:\n{message}')


def preprocess_epiphan_image(image):
    image = np.rot90(np.transpose(image, (1,2,0)), 2)
    if image.shape[2] == 1:
        image = np.concatenate([image, image, image], axis=2)
    return image


if __name__ == "__main__":
    args = parse_args()
    run_client(args)
