
import argparse
import csv
import time
import torch
import monai
import statistics

from pathlib import Path

from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric
from monai.data import DataLoader
from monai.transforms import Compose, Activations, AsDiscrete
from UltrasoundDataset import UltrasoundDataset

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

    # Confusion matrix labels
    cm_labels_dict = {
        "accuracy": 0,
        "precision": 1,
        "sensitivity": 2,
        "specificity": 3,
        "f1_score": 4
    }

    # Build a list of labels for the confusion matrix
    cm_labels = [label for label in cm_labels_dict]

    # Metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    confusion_matrix_metric = ConfusionMatrixMetric(
        include_background=True, 
        metric_name=cm_labels,
        reduction="mean"
    )

    # Test loop
    with torch.no_grad():
        inference_times = []

        for test_data in test_loader:
            inputs, labels = test_data["image"].to(device), test_data["label"].to(device)
            inputs = inputs.float()
            inputs = inputs.permute(0, 3, 1, 2)
            labels = labels.permute(0, 3, 1, 2)

            start_time = time.time()
            
            outputs = model(inputs)
            outputs = Activations(sigmoid=True)(outputs)
            outputs = torch.argmax(outputs, dim=1, keepdim=True)
            outputs = AsDiscrete(threshold_values=True)(outputs)
            
            elapsed_time = time.time() - start_time
            inference_times.append(elapsed_time)
            
            dice_metric(y_pred=outputs, y=labels)
            iou_metric(y_pred=outputs, y=labels)
            confusion_matrix_metric(y_pred=outputs, y=labels)

    print(f"Dice Score: {dice_metric.aggregate().item():.3f}")
    print(f"IoU: {iou_metric.aggregate().item():.3f}")
    confusion_matrix_aggregate = confusion_matrix_metric.aggregate()

    # Print the confusion matrix
    print("\nConfusion matrix metrics:")
    for label in cm_labels:
        print(f"    {label}: {confusion_matrix_aggregate[cm_labels_dict[label]].item():.3f}")

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
        "dice_score": dice_metric.aggregate().item(),
        "iou": iou_metric.aggregate().item(),
        "accuracy": confusion_matrix_aggregate[cm_labels_dict["accuracy"]].item(),
        "precision": confusion_matrix_aggregate[cm_labels_dict["precision"]].item(),
        "sensitivity": confusion_matrix_aggregate[cm_labels_dict["sensitivity"]].item(),
        "specificity": confusion_matrix_aggregate[cm_labels_dict["specificity"]].item(),
        "f1_score": confusion_matrix_aggregate[cm_labels_dict["f1_score"]].item(),
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
