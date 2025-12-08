import argparse
import traceback
import sys
from pathlib import Path

import torch
import wandb
import yaml
from tqdm.auto import tqdm
from ultralytics import YOLO


# Parse command line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--data-yaml", type=str, default='D:/Repos/aigt/UltrasoundObjectDetection/YOLOv8/lung_us.yml')
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--k-folds", type=int, default=4)
    try:
        return parser.parse_args()
    except SystemExit as err:
        traceback.print_exc()
        sys.exit(err.code)


def _resolve_folds_root_and_stem(data_yaml: str) -> tuple[Path, str]:
    data_yaml_path = Path(data_yaml).expanduser().resolve()
    with data_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = Path(data["path"]).expanduser().resolve()
    folds_root = dataset_root / "kfold_splits"
    return folds_root, data_yaml_path.stem


def kfold_cross_validate(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_id = wandb.util.generate_id()

    folds_root, yaml_stem = _resolve_folds_root_and_stem(args.data_yaml)

    for fold_idx in tqdm(range(1, args.k_folds + 1)):
        fold_dir = folds_root / f"fold_{fold_idx}"
        fold_yaml = fold_dir / f"{yaml_stem}_fold_{fold_idx}.yml"

        if not fold_yaml.exists():
            raise FileNotFoundError(
                f"Fold dataset YAML not found for fold {fold_idx}: {fold_yaml}"
            )

        model = YOLO(model=args.weights)
        model.train(device=device,
                    workers=1,
                    data=str(fold_yaml),
                    epochs=args.epochs,
                    imgsz=args.image_size,
                    batch=args.batch,
                    project='BLUE Protocol - Object Detection',
                    name=f'{experiment_id}_{args.k_folds}_fold_cross_validation_fold#{fold_idx}')


if __name__ == "__main__":
    args = parse_args()
    kfold_cross_validate(args)
