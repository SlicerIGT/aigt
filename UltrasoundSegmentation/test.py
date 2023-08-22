import tqdm
import argparse
import csv
import time
import torch
import monai
import statistics
from pathlib import Path

from monai.data import DataLoader
from monai.transforms import Compose, Activations, AsDiscrete
from UltrasoundDataset import UltrasoundDataset

import metrics as fm


def main(args):
    # Ensure reproducibility
    monai.utils.set_determinism(seed=42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.model_path).to(device)
    model.eval()

    # Transforms for test dataset
    test_transforms = Compose([
        # Assuming the transforms for the test dataset are the same as the validation dataset in train.py
        # Add transforms here later if needed
    ])

    # Create test dataset and dataloader
    batch_size = 1
    test_ds = UltrasoundDataset(args.test_data_path, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    num_test_batches = len(test_loader)

    # Test loop
    with torch.no_grad():
        inference_times = []
        avg_acc = 0
        avg_pre = 0
        avg_sen = 0
        avg_spe = 0
        avg_f1 = 0
        avg_dice = 0
        avg_iou = 0

        for test_data in tqdm.tqdm(test_loader):
            inputs, labels = test_data["image"].to(device), test_data["label"].to(device)
            inputs = inputs.float()
            inputs = inputs.permute(0, 3, 1, 2)
            labels = labels.permute(0, 3, 1, 2)
            labels = monai.networks.one_hot(labels, num_classes=inputs.shape[1])

            start_time = time.time()
            
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            
            elapsed_time = time.time() - start_time
            inference_times.append(elapsed_time)
            
            avg_acc += fm.fuzzy_accuracy(pred=outputs, target=labels, sigmoid=True) / num_test_batches
            avg_pre += fm.fuzzy_precision(pred=outputs, target=labels, sigmoid=True) / num_test_batches
            avg_sen += fm.fuzzy_sensitivity(pred=outputs, target=labels, sigmoid=True) / num_test_batches
            avg_spe += fm.fuzzy_specificity(pred=outputs, target=labels, sigmoid=True) / num_test_batches
            avg_f1 += fm.fuzzy_f1_score(pred=outputs, target=labels, sigmoid=True) / num_test_batches
            avg_dice += fm.fuzzy_dice(pred=outputs, target=labels, sigmoid=True) / num_test_batches
            avg_iou += fm.fuzzy_iou(pred=outputs, target=labels, sigmoid=True) / num_test_batches
    
    # Printing metrics
    print("\nPerformance metrics:")
    print(f"    Accuracy:    {avg_acc:.3f}")
    print(f"    Precision:   {avg_pre:.3f}")
    print(f"    Sensitivity: {avg_sen:.3f}")
    print(f"    Specificity: {avg_spe:.3f}")
    print(f"    F1 score:    {avg_f1:.3f}")
    print(f"    Dice score:  {avg_dice:.3f}")
    print(f"    IoU:         {avg_iou:.3f}")

    # Printing performance statistics
    print("\nPerformance statistics:")
    print(f"    Number of test images:   {len(test_ds)}")
    print(f"    Median inference time:   {statistics.median(inference_times):.3f} seconds")
    print(f"    Median FPS:              {1 / statistics.median(inference_times):.3f}")
    print(f"    Average inference time:  {statistics.mean(inference_times):.3f} seconds")
    print(f"    Inference time SD:       {statistics.stdev(inference_times):.3f} seconds")
    print(f"    Maximum time:            {max(inference_times):.3f} seconds")
    print(f"    Minimum time:            {min(inference_times):.3f} seconds")

    # Put all metrics into a dictionary so it can be written to a CSV file later
    metrics_dict = {
        "dice_score": avg_dice,
        "iou": avg_iou,
        "accuracy": avg_acc,
        "precision": avg_pre,
        "sensitivity": avg_sen,
        "specificity": avg_spe,
        "f1_score": avg_f1,
        "num_test_images": len(test_ds),
        "median_inference_time": statistics.median(inference_times),
        "median_fps": 1 / statistics.median(inference_times),
        "average_inference_time": statistics.mean(inference_times),
        "inference_time_sd": statistics.stdev(inference_times),
        "maximum_time": max(inference_times),
        "minimum_time": min(inference_times)
    }

    # Create the CSV file and its path if it doesn't exist
    Path(args.output_csv_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Write the metrics to a CSV file.
    with open(args.output_csv_file, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=metrics_dict.keys())
        writer.writeheader()
        writer.writerow(metrics_dict)

    print("\nMetrics written to CSV file: " + args.output_csv_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained segmentation model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--output_csv_file", type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset already in slices format.")
    args = parser.parse_args()
    main(args)
