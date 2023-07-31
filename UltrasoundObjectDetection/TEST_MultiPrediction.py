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
from pathlib import Path
import time
import pyigtl
from YOLOv5.model import ObjectDetectionModel
from threading import Thread
from queue import SimpleQueue
import torch


ROOT = Path(__file__).parent.resolve()

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="YOLOv5/lung_us_pretrained.torchscript")
    parser.add_argument("--data-yaml", type=str, default="YOLOv5/lung_us.yml")
    parser.add_argument("--input-device-name", type=str, default="Image_Reference")
    parser.add_argument("--output-device-name", type=str, default="Inference")
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--line-thickness", type=int, default=2)
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
def predict_primary_signs(args):

    model = None

    while True:
        message = input_client.wait_for_message(args.input_device_name, timeout=-1)

        if isinstance(message, pyigtl.ImageMessage):
            if model is None:
                input_size = message.image.shape[1:3]
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model_path = args.model if Path(args.model).is_absolute() else f'{str(ROOT)}/{args.model}'
                data_yaml_path = args.data_yaml if Path(args.data_yaml).is_absolute() else f'{str(ROOT)}/{args.data_yaml}'
                model = ObjectDetectionModel(model=model_path,
                                             data_yaml=data_yaml_path,
                                             device=device,
                                             line_thickness=args.line_thickness,
                                             input_size=input_size,
                                             target_size=args.target_size)

            image = preprocess_epiphan_image(message.image)

            prediction = model.predict(image, args.confidence_threshold)
            image_message = pyigtl.ImageMessage(np.flip(np.flip(prediction, axis=1), axis=2), device_name=args.output_device_name)
            output_server.send_message(image_message, wait=True)
        else:
            print(f'Unexpected message format. Message:\n{message}')


def predict_lung_sliding(args):
    model = None
    while True:
        line_control_points = input_client.wait_for_message("mmode_line_points", timeout=-1)
        if isinstance(line_control_points, pyigtl.PointMessage):
            if model is None:
                pass #TODO: instantiate PTX classification model
            
            print('triggered')


def collect_frames(n_seconds):
    start_time = time.time()
    while time.time() < start_time + n_seconds:



def preprocess_epiphan_image(image):
    image = np.rot90(np.transpose(image, (1,2,0)), 2)
    if image.shape[2] == 1:
        image = np.concatenate([image, image, image], axis=2)
    return image


if __name__ == "__main__":
    args = parse_args()
    input_client = pyigtl.OpenIGTLinkClient(host=args.host, port=args.input_port)
    output_server = pyigtl.OpenIGTLinkServer(port=args.output_port)

    lung_sliding_detection = Thread(target=predict_lung_sliding, args=(args,))
    
    stationary_sign_detection = Thread(target=predict_primary_signs, args=(args,))    

    
    lung_sliding_detection.start()
    stationary_sign_detection.start()


def onInputModified(volumeNode, event):
    print(f'inputnode {volumeNode.GetName()} modified')


