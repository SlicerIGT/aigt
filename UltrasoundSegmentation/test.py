import tqdm
import argparse
import time
import os
import yaml
import torch
import monai
import statistics
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric
from monai.data import DataLoader
from monai.transforms import Compose, AsDiscrete
from datasets import UltrasoundDataset

from metrics import FuzzyMetrics


LIMIT_TEST_BATCHES = None # Make this None to process all test batches


def test_model(model_path: str,
               test_data_path: str,
               output_csv_file: str,
               num_sample_images: int = 10,
               output_dir: str = "output"):
    # Ensure reproducibility
    monai.utils.set_determinism(seed=42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path).to(device)
    model.eval()

    # Get number of output channels from config file in model folder
    with open(Path(model_path).parent / "train_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    num_classes = config["out_channels"]

    # Transforms for test dataset
    test_transforms = Compose([
        # Assuming the transforms for the test dataset are the same as the validation dataset in train.py
        # Add transforms here later if needed
    ])

    # Create test dataset and dataloader
    test_ds = UltrasoundDataset(test_data_path, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Initialize crisp metrics
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
    dice_metric = DiceMetric(include_background=True)
    iou_metric = MeanIoU(include_background=True)
    confusion_matrix_metric = ConfusionMatrixMetric(
        include_background=True, 
        metric_name=cm_labels,
        compute_sample=True
    )

    # Initialize metrics
    fm = FuzzyMetrics(num_classes=num_classes)

    # Take a sample image and matching segmentation from the test dataset and print the data shapes and value ranges
    sample_data = test_ds[0]
    sample_image = sample_data["image"]
    sample_label = sample_data["label"]
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample image value range: {sample_image.min()} to {sample_image.max()}")
    print(f"Sample label shape: {sample_label.shape}")
    print(f"Sample label value range: {sample_label.min()} to {sample_label.max()}")

    # Generate a list of random indices for sample images to save
    sample_indices = torch.randint(0, len(test_ds), (num_sample_images,))
    sample_indices = sample_indices.tolist()

    if LIMIT_TEST_BATCHES is not None:
        print(f"\nLimiting test batches to {LIMIT_TEST_BATCHES} batches.")
        sample_indices = torch.randint(0, LIMIT_TEST_BATCHES, (num_sample_images,))
        sample_indices = sample_indices.tolist()

    # Create output directory if it doesn't already exist
    if num_sample_images > 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create plot for sample images
    if num_sample_images > 0:
        fig, axs = plt.subplots(num_sample_images, 3, figsize=(15, 15 * num_sample_images / 3))
        plot_idx = 0

    # Test loop
    with torch.inference_mode():
        inference_times = []

        # Make sure we have an index that increments by one with each iteration
        for batch_index, test_data in enumerate(tqdm.tqdm(test_loader)):
            inputs, labels = test_data["image"].to(device), test_data["label"].to(device)
            inputs = inputs.float()
            inputs = inputs.permute(0, 3, 1, 2)
            labels = labels.permute(0, 3, 1, 2)
            labels = monai.networks.one_hot(labels, num_classes=num_classes)

            if LIMIT_TEST_BATCHES is not None:
                print(f"\ninputs shape:        {inputs.shape}")
                print(f"inputs value range:  {inputs.min()} to {inputs.max()}")
                print(f"labels shape:        {labels.shape}")
                print(f"labels value range:  {labels.min()} to {labels.max()}")
            
            start_time = time.time()

            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            
            elapsed_time = time.time() - start_time
            inference_times.append(elapsed_time)
            
            if LIMIT_TEST_BATCHES is not None:
                print(f"outputs shape:       {outputs.shape}")
                print(f"outputs value range: {outputs.min()} to {outputs.max()}")
            
            # TODO: can remove this once debugging is done and change softmax arg to True below
            outputs = torch.softmax(outputs, dim=1)

            # Threshold output to get binary labels
            outputs_crisp = AsDiscrete(threshold=0.5)(outputs)

            # Update crisp metrics
            dice_metric(y_pred=outputs_crisp, y=labels)
            iou_metric(y_pred=outputs_crisp, y=labels)
            confusion_matrix_metric(y_pred=outputs_crisp, y=labels)

            # Update fuzzy metrics
            fm.update_metrics(outputs, labels, softmax=False)

            if LIMIT_TEST_BATCHES is not None:
                print(f"outputs shape:       {outputs.shape}")
                print(f"outputs value range: {outputs.min()} to {outputs.max()}")

            # If this is a sample image, save the input image and the output image as png files
            if batch_index in sample_indices:
                # Save input image
                input_image = inputs[0].permute(1, 2, 0).cpu().numpy()
                if input_image.max() <= 1.0:
                    input_image = input_image * 255
                input_image = input_image.astype("uint8")
                input_image = input_image[:, :, 0]
                input_image = np.flip(input_image, axis=0)  # Flip the image vertically
                input_image_pil = Image.fromarray(input_image)
                input_image_pil.save(Path(output_dir) / f"{batch_index:04}_input.png")

                # Save labels
                label_image = labels[0].permute(1, 2, 0).cpu().numpy()
                label_image = (1.0 - label_image) * 255
                label_image = label_image.astype("uint8")
                label_image = label_image[:, :, 0]
                label_image = np.flip(label_image, axis=0)
                label_image_pil = Image.fromarray(label_image)
                label_image_pil.save(Path(output_dir) / f"{batch_index:04}_label.png")

                # Save output image
                output_image = outputs[0].permute(1, 2, 0).cpu().numpy()
                output_image = (1.0 - output_image) * 255  # Invert the background, which results in the sum of all labels
                output_image = output_image.astype("uint8")
                output_image = output_image[:, :, 0]
                output_image = np.flip(output_image, axis=0)
                output_image_pil = Image.fromarray(output_image)
                output_image_pil.save(Path(output_dir) / f"{batch_index:04}_output.png")

                # Plot the input, label, and output images
                if plot_idx == 0:
                    axs[plot_idx, 0].set_title("Input")
                    axs[plot_idx, 1].set_title("Label")
                    axs[plot_idx, 2].set_title("Output")
                axs[plot_idx, 0].imshow(input_image, cmap="gray")
                axs[plot_idx, 1].imshow(label_image, cmap="gray")
                axs[plot_idx, 2].imshow(output_image, cmap="gray")
                plot_idx += 1

            # Limit the number of test batches to process
            if LIMIT_TEST_BATCHES is not None:
                if len(inference_times) >= LIMIT_TEST_BATCHES:
                    break

    # Save the plot of sample images
    if num_sample_images > 0:
        fig.savefig(Path(output_dir) / "sample_images.png")

    # Aggregate binary metrics by class
    cm_aggregate_batch = confusion_matrix_metric.aggregate(reduction="mean_batch")
    avg_acc_cls = cm_aggregate_batch[cm_labels_dict["accuracy"]].cpu().numpy()
    avg_pre_cls = cm_aggregate_batch[cm_labels_dict["precision"]].cpu().numpy()
    avg_sen_cls = cm_aggregate_batch[cm_labels_dict["sensitivity"]].cpu().numpy()
    avg_spe_cls = cm_aggregate_batch[cm_labels_dict["specificity"]].cpu().numpy()
    avg_f1_cls = cm_aggregate_batch[cm_labels_dict["f1_score"]].cpu().numpy()
    avg_dice_cls = dice_metric.aggregate(reduction="mean_batch").cpu().numpy()
    avg_iou_cls = iou_metric.aggregate(reduction="mean_batch").cpu().numpy()

    # Aggregate binary metrics with average
    cm_aggregate = confusion_matrix_metric.aggregate(reduction="mean")
    avg_acc = cm_aggregate[cm_labels_dict["accuracy"]].item()
    avg_pre = cm_aggregate[cm_labels_dict["precision"]].item()
    avg_sen = cm_aggregate[cm_labels_dict["sensitivity"]].item()
    avg_spe = cm_aggregate[cm_labels_dict["specificity"]].item()
    avg_f1 = cm_aggregate[cm_labels_dict["f1_score"]].item()
    avg_dice = dice_metric.aggregate(reduction="mean").item()
    avg_iou = iou_metric.aggregate(reduction="mean").item()

    # Aggregate fuzzy metrics
    avg_acc_f, avg_pre_f, avg_sen_f, avg_spe_f, avg_f1_f, avg_dice_f, avg_iou_f = fm.get_total_mean_metrics()
    
    # Printing metrics
    print("\nCrisp Performance metrics:")
    print(f"    Accuracy:    {avg_acc:.3f}")
    print(f"    Precision:   {avg_pre:.3f}")
    print(f"    Sensitivity: {avg_sen:.3f}")
    print(f"    Specificity: {avg_spe:.3f}")
    print(f"    F1 score:    {avg_f1:.3f}")
    print(f"    Dice score:  {avg_dice:.3f}")
    print(f"    IoU:         {avg_iou:.3f}")

    print("\nFuzzy Performance metrics:")
    print(f"    Accuracy:    {avg_acc_f:.3f}")
    print(f"    Precision:   {avg_pre_f:.3f}")
    print(f"    Sensitivity: {avg_sen_f:.3f}")
    print(f"    Specificity: {avg_spe_f:.3f}")
    print(f"    F1 score:    {avg_f1_f:.3f}")
    print(f"    Dice score:  {avg_dice_f:.3f}")
    print(f"    IoU:         {avg_iou_f:.3f}")

    # Printing performance statistics
    print("\nPerformance statistics:")
    print(f"    Number of test images:   {len(test_ds)}")
    print(f"    Median inference time:   {statistics.median(inference_times):.3f} seconds")
    print(f"    Median FPS:              {1 / statistics.median(inference_times):.3f}")
    print(f"    Average inference time:  {statistics.mean(inference_times):.3f} seconds")
    print(f"    Inference time SD:       {statistics.stdev(inference_times):.3f} seconds")
    print(f"    Maximum time:            {max(inference_times):.3f} seconds")
    print(f"    Minimum time:            {min(inference_times):.3f} seconds")

    # Create Pandas dataframe for metrics
    metrics_df = fm.get_metrics_as_dataframe()
    metrics_df.loc["accuracy"] = np.append(avg_acc_cls, avg_acc)
    metrics_df.loc["precision"] = np.append(avg_pre_cls, avg_pre)
    metrics_df.loc["sensitivity"] = np.append(avg_sen_cls, avg_sen)
    metrics_df.loc["specificity"] = np.append(avg_spe_cls, avg_spe)
    metrics_df.loc["f1_score"] = np.append(avg_f1_cls, avg_f1)
    metrics_df.loc["dice"] = np.append(avg_dice_cls, avg_dice)
    metrics_df.loc["iou"] = np.append(avg_iou_cls, avg_iou)

    # Add performance metrics
    metrics_df.loc["num_test_images"] = np.repeat(len(test_ds), len(metrics_df.columns))
    metrics_df.loc["median_inference_time"] = np.repeat(statistics.median(inference_times), len(metrics_df.columns))
    metrics_df.loc["median_fps"] = np.repeat(np.float32(1 / statistics.median(inference_times)), len(metrics_df.columns))
    metrics_df.loc["average_inference_time"] = np.repeat(np.float32(statistics.mean(inference_times)), len(metrics_df.columns))
    metrics_df.loc["inference_time_sd"] = np.repeat(np.float32(statistics.stdev(inference_times)), len(metrics_df.columns))
    metrics_df.loc["maximum_time"] = np.repeat(np.float32(max(inference_times)), len(metrics_df.columns))
    metrics_df.loc["minimum_time"] = np.repeat(np.float32(min(inference_times)), len(metrics_df.columns))

    # Make sure only forward slashes are used in the file paths
    output_csv_file = output_csv_file.replace("\\", "/")
    
    # Create the CSV folder path if it doesn't exist and save
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    metrics_df.to_csv(output_csv_file, index_label="class")

    print("\nMetrics written to CSV file: " + str(output_csv_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained segmentation model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset already in slices format.")
    parser.add_argument("--output_csv_file", type=str, default="test_results.csv", help="Path to the output CSV file.")
    parser.add_argument("--num_sample_images", type=int, default=10, help="Number of sample images to save in the output folder.")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to the output folder.")
    args = parser.parse_args()
    test_model(args.model_path, args.test_data_path, args.output_csv_file, args.num_sample_images, args.output_dir)
