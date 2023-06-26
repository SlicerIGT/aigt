"""
Implements an OpenIGTLink client that expect pyigtl.ImageMessage and returns pyigtl.ImageMessage with YOLOv5 inference added to the image.
"""

import argparse
import traceback
import sys
import numpy as np
import pyigtl
import numpy as np
from YOLOv5 import YOLOv5
import torch

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str)
    parser.add_argument("--input-device-name", type=str, default="Image_Reference")
    parser.add_argument("--output-device-name", type=str, default="Inference")
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18944)
    try:
        return parser.parse_args()
    except SystemExit as err:
        traceback.print_exc()
        sys.exit(err.code)


def main(args):
    client = pyigtl.OpenIGTLinkClient(host=args.host, port=args.port)
    model = None

    while True:
        message = client.wait_for_message(args.input_device_name, timeout=3)
        if message:
            if model is None:
                input_size = message.image.shape[1:3]
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = YOLOv5(weights=args.weights,
                               device=device,
                               line_thickness=args.line_thickness,
                               input_size=input_size,
                               target_size=args.target_size)

            image = np.rot90(np.transpose(message.image, (1,2,0)), 2)

            prediction = model.predict(image, args.confidence_threshold)
            image_message = pyigtl.ImageMessage(np.flip(np.flip(prediction, axis=1), axis=2), device_name=args.output_device_name)
            client.send_message(image_message, wait=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
