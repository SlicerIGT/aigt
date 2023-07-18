import argparse
import traceback
import sys
import torch
from ultralytics import YOLO


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8s.pt")
    parser.add_argument("--data-yaml", type=str, default='D:/Data/Lung/ObjectDetection/training.yaml')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    try:
        return parser.parse_args()
    except SystemExit as err:
        traceback.print_exc()
        sys.exit(err.code)


# runs the client in an infinite loop, waiting for messages from the server. Once a message is received,
# the message is processed and the inference is sent back to the server as a pyigtl ImageMessage.
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(model=args.weights)

    model.train(data=args.data_yaml,
                epochs=args.epochs,
                imgsz=args.image_size,
                batch=args.batch)



if __name__ == "__main__":
    args = parse_args()
    train(args)
