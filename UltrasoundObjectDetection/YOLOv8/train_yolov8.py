import argparse
import traceback
import sys
import torch
import wandb
from ultralytics import YOLO


# Parse command line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--data-yaml", type=str, default='D:/Repos/aigt/UltrasoundObjectDetection/YOLOv8/lung_us.yml')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    try:
        return parser.parse_args()
    except SystemExit as err:
        traceback.print_exc()
        sys.exit(err.code)


def train(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_id = wandb.util.generate_id()
    model = YOLO(model=args.weights)

    model.train(device=device,
                workers=1,
                data=args.data_yaml,
                epochs=args.epochs,
                imgsz=args.image_size,
                batch=args.batch,
                project='BLUE Protocol - Object Detection',
                name=f'{experiment_id}_full_training')



if __name__ == "__main__":
    args = parse_args()
    train(args)
