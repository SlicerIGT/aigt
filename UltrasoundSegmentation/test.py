import tqdm
import argparse
import datetime
import time
import yaml
import torch
import monai
import statistics
from pathlib import Path
from PIL import Image

from monai.data import DataLoader
from monai.transforms import Compose, Activations, AsDiscrete
from UltrasoundDataset import UltrasoundDataset

from metrics import FuzzyMetrics


LIMIT_TEST_BATCHES = 50  # Make this None to process all test batches


def main(args):
    # Ensure reproducibility
    monai.utils.set_determinism(seed=42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.model_path).to(device)
    model.eval()

    # Get number of output channels from config file in model folder
    with open(Path(args.model_path).parent / "train_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    num_classes = config["out_channels"]

    # Transforms for test dataset
    test_transforms = Compose([
        # Assuming the transforms for the test dataset are the same as the validation dataset in train.py
        # Add transforms here later if needed
    ])

    # Create test dataset and dataloader
    test_ds = UltrasoundDataset(args.test_data_path, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

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
    sample_indices = torch.randint(0, len(test_ds), (args.num_sample_images,))
    sample_indices = sample_indices.tolist()

    if LIMIT_TEST_BATCHES is not None:
        print(f"\nLimiting test batches to {LIMIT_TEST_BATCHES} batches.")
        sample_indices = torch.randint(0, LIMIT_TEST_BATCHES, (args.num_sample_images,))
        sample_indices = sample_indices.tolist()

    # Create output directory if it doesn't already exist
    if args.num_sample_images > 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

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

        # Make sure we have an index that increments by one with each iteration
        test_loader = enumerate(test_loader)
        for batch_index, test_data in tqdm.tqdm(test_loader):
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

            # Update metrics
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
                input_image = Image.fromarray(input_image)
                input_image.save(Path(args.output_dir) / f"{batch_index:04}_input.png")

                # Save labels
                label_image = labels[0].permute(1, 2, 0).cpu().numpy()
                label_image = (1.0 - label_image) * 255
                label_image = label_image.astype("uint8")
                label_image = label_image[:, :, 0]
                label_image = Image.fromarray(label_image)
                label_image.save(Path(args.output_dir) / f"{batch_index:04}_label.png")

                # Save output image
                output_image = outputs[0].permute(1, 2, 0).cpu().numpy()
                output_image = (1.0 - output_image) * 255  # Invert the background, which results in the sum of all labels
                output_image = output_image.astype("uint8")
                output_image = output_image[:, :, 0]
                output_image = Image.fromarray(output_image)
                output_image.save(Path(args.output_dir) / f"{batch_index:04}_output.png")

            # Limit the number of test batches to process
            if LIMIT_TEST_BATCHES is not None:
                if len(inference_times) >= LIMIT_TEST_BATCHES:
                    break

    # Aggregate metrics
    avg_acc, avg_pre, avg_sen, avg_spe, avg_f1, avg_dice, avg_iou = fm.get_total_mean_metrics()
    
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

    # Create Pandas dataframe for metrics
    metrics_df = fm.get_metrics_as_dataframe()
    metrics_df.at["num_test_images", "total"] = len(test_ds)
    metrics_df.at["median_inference_time", "total"] = statistics.median(inference_times)
    metrics_df.at["median_fps", "total"] = 1 / statistics.median(inference_times)
    metrics_df.at["average_inference_time", "total"] = statistics.mean(inference_times)
    metrics_df.at["inference_time_sd", "total"] = statistics.stdev(inference_times)
    metrics_df.at["maximum_time", "total"] = max(inference_times)
    metrics_df.at["minimum_time", "total"] = min(inference_times)

    # Create the CSV folder path if it doesn't exist and save
    output_csv_file = Path(args.output_dir) / f"test_results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    output_csv_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_csv_file)

    print("\nMetrics written to CSV file: " + str(output_csv_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained segmentation model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset already in slices format.")
    parser.add_argument("--num_sample_images", type=int, default=10, help="Number of sample images to save in the output folder.")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to the output folder.")
    args = parser.parse_args()
    main(args)
