"""
This script tests multiple trained segmentation models on a test dataset.
The input CSV file should have the following columns:
    model_path: Path to the trained model
    output_csv_file: Path to the output CSV file
    output_dir: Path to the output folder
m
"""

import csv
import argparse
from test import test_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test multiple trained segmentation models.")
    parser.add_argument("--models_csv", type=str, required=True, help="Path to the CSV file with model information.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset already in slices format.")
    parser.add_argument("--num_sample_images", type=int, default=10, help="Number of sample images to save in the output folder.")
    args = parser.parse_args()

    with open(args.models_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            model_path = row['model_path']
            output_csv_file = row['output_csv_file']
            output_dir = row['output_dir']
            test_model(model_path, args.test_data_path, output_csv_file, args.num_sample_images, output_dir)
