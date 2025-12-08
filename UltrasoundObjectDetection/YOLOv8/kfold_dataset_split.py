import argparse
import random
import sys
from pathlib import Path
from typing import List
import shutil

import yaml


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create patient-wise train/val folds from a YOLO dataset YAML definition.",
    )
    parser.add_argument(
        "--data-yaml",
        type=str,
        default="lung_us.yml",
        help="Path to YOLO dataset YAML (e.g. lung_us.yml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store K-fold split files (defaults to <data-yaml-dir>/kfold_splits).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before splitting.",
    )

    try:
        return parser.parse_args()
    except SystemExit as err:
        # Mirror style used in train_yolov8.py
        raise


def load_dataset_images(data_yaml_path: Path) -> List[Path]:
    with data_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    base_path = Path(data["path"]).expanduser()

    image_dirs: List[Path] = []
    train = data.get("train")
    val = data.get("val")

    if isinstance(train, str) and train:
        image_dirs.append(base_path / train)
    if isinstance(val, str) and val:
        image_dirs.append(base_path / val)

    images: List[Path] = []
    for img_dir in image_dirs:
        if not img_dir.exists():
            continue
        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
                images.append(p.resolve())

    if not images:
        raise RuntimeError(f"No images found using paths derived from {data_yaml_path}.")

    images.sort()
    return images


def split_indices_kfold(n_items: int, k_folds: int) -> List[List[int]]:
    if k_folds <= 1:
        raise ValueError("k_folds must be >= 2 for cross-validation.")
    if n_items < k_folds:
        raise ValueError(
            f"Number of items ({n_items}) must be >= number of folds ({k_folds})."
        )

    fold_sizes = [n_items // k_folds] * k_folds
    remainder = n_items % k_folds
    for i in range(remainder):
        fold_sizes[i] += 1

    folds: List[List[int]] = []
    current = 0
    for size in fold_sizes:
        folds.append(list(range(current, current + size)))
        current += size

    return folds


def _patient_id_from_image_path(image_path: Path) -> str:
    """Extract patient ID from image filename.

    Assumes filenames start with the patient ID followed by an underscore,
    e.g. "SCN22_frame001.png" -> patient ID "SCN22".
    """
    stem = image_path.stem
    return stem.split("_")[0]


def build_patient_folds(images: List[Path], rng: random.Random) -> List[List[int]]:
    """Create K folds where each patient appears in exactly one fold.

    The number of folds is effectively capped at the number of distinct
    patients. Patients are distributed across folds to keep the number
    of images per fold roughly balanced.
    """
    patient_to_indices: dict[str, List[int]] = {}
    for idx, img_path in enumerate(images):
        patient_id = _patient_id_from_image_path(img_path)
        patient_to_indices.setdefault(patient_id, []).append(idx)

    patient_ids = list(patient_to_indices.keys())
    if not patient_ids:
        raise RuntimeError("No patient IDs could be inferred from image filenames.")

    n_patients = len(patient_ids)
    if n_patients < 2:
        raise ValueError("Need at least 2 distinct patients for splitting.")

    rng.shuffle(patient_ids)

    # One fold per patient: maximum number of folds.
    folds: List[List[int]] = []
    for pid in patient_ids:
        fold_indices = list(patient_to_indices[pid])
        fold_indices.sort()
        folds.append(fold_indices)

    return folds


def _label_path_for_image(image_path: Path) -> Path:
    if image_path.parent.name == "images":
        split_root = image_path.parent.parent
        labels_dir = split_root / "labels"
        return labels_dir / (image_path.stem + ".txt")
    return image_path.with_suffix(".txt")


def write_folds_to_directories(
    images: List[Path],
    folds: List[List[int]],
    output_dir: Path,
    original_yaml_data: dict,
    yaml_stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    names = original_yaml_data.get("names")

    for fold_idx, val_indices in enumerate(folds):
        fold_number = fold_idx + 1
        val_set = {idx for idx in val_indices}

        train_paths = [images[i] for i in range(len(images)) if i not in val_set]
        val_paths = [images[i] for i in val_indices]

        fold_dir = output_dir / f"fold_{fold_number}"
        train_images_dir = fold_dir / "train" / "images"
        train_labels_dir = fold_dir / "train" / "labels"
        val_images_dir = fold_dir / "val" / "images"
        val_labels_dir = fold_dir / "val" / "labels"

        train_images_dir.mkdir(parents=True, exist_ok=True)
        train_labels_dir.mkdir(parents=True, exist_ok=True)
        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)

        for img_path in train_paths:
            dst_img = train_images_dir / img_path.name
            shutil.copy2(img_path, dst_img)

            label_src = _label_path_for_image(img_path)
            if label_src.exists():
                dst_label = train_labels_dir / label_src.name
                shutil.copy2(label_src, dst_label)

        for img_path in val_paths:
            dst_img = val_images_dir / img_path.name
            shutil.copy2(img_path, dst_img)

            label_src = _label_path_for_image(img_path)
            if label_src.exists():
                dst_label = val_labels_dir / label_src.name
                shutil.copy2(label_src, dst_label)

        fold_yaml = fold_dir / f"{yaml_stem}_fold_{fold_number}.yml"
        fold_yaml_data = {
            "path": str(fold_dir),
            "train": "train/images",
            "val": "val/images",
        }
        if names is not None:
            fold_yaml_data["names"] = names

        with fold_yaml.open("w", encoding="utf-8") as f_yaml:
            yaml.safe_dump(fold_yaml_data, f_yaml, sort_keys=False)


def main() -> None:
    args = parse_args()

    data_yaml_path = Path(args.data_yaml).expanduser()
    if not data_yaml_path.is_absolute():
        data_yaml_path = data_yaml_path.resolve()

    if not data_yaml_path.exists():
        script_dir = Path(__file__).resolve().parent
        candidate = script_dir / args.data_yaml
        if candidate.exists():
            data_yaml_path = candidate
        else:
            raise FileNotFoundError(
                f"Dataset YAML file not found. Tried: {data_yaml_path} and {candidate}"
            )

    with data_yaml_path.open("r", encoding="utf-8") as f:
        original_yaml_data = yaml.safe_load(f)

    if args.output_dir is None:
        dataset_root = Path(original_yaml_data["path"]).expanduser().resolve()
        output_dir = dataset_root / "kfold_splits"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    images = load_dataset_images(data_yaml_path)

    rng = random.Random(args.seed)
    # Build folds at patient level so that all images from a patient
    # are always in the same fold. One fold per patient.
    folds = build_patient_folds(images, rng)
    write_folds_to_directories(
        images,
        folds,
        output_dir,
        original_yaml_data,
        yaml_stem=data_yaml_path.stem,
    )

    print(f"Created {len(folds)}-fold directory splits for {len(images)} images.")
    print(f"Fold folders and YAML files are in: {output_dir}")


if __name__ == "__main__":
    main()
