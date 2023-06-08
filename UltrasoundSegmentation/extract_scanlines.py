import os
import glob
import argparse
import logging
import numpy as np
import yaml
from tqdm import tqdm
from scipy.ndimage import map_coordinates



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--scanconvert-config", type=str)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-file", type=str)
    return parser.parse_args()


def scan_conversion_inverse(scanconversion_config):
    """
    Compute cartesian coordianates for inverse scan conversion.
    Mapping from curvilinear image to a rectancular image of scan lines as columns.
    The returned cartesian coordinates can be used to map the curvilinear image to a rectangular image using scipy.ndimage.map_coordinates.

    Args:
        scanconversion_config (dict): Dictionary with scan conversion parameters.

    Rerturns:
        x_cart (np.ndarray): x coordinates of the cartesian grid.
        y_cart (np.ndarray): y coordinates of the cartesian grid.

    Example:
        >>> x_cart, y_cart = scan_conversion_inverse(scanconversion_config)
        >>> scan_converted_image = map_coordinates(ultrasound_data[0, :, :, 0], [x_cart, y_cart], order=3, mode="nearest")
        >>> scan_converted_segmentation = map_coordinates(segmentation_data[0, :, :, 0], [x_cart, y_cart], order=0, mode="nearest")
    """

    # Create sampling points in polar coordinates

    initial_radius = np.deg2rad(scanconversion_config["angle_min_degrees"])
    final_radius = np.deg2rad(scanconversion_config["angle_max_degrees"])
    radius_start_px = scanconversion_config["radius_start_pixels"]
    radius_end_px = scanconversion_config["radius_end_pixels"]

    theta, r = np.meshgrid(np.linspace(initial_radius, final_radius, scanconversion_config["num_samples_along_lines"]),
                           np.linspace(radius_start_px, radius_end_px, scanconversion_config["num_lines"]))

    # Convert the polar coordinates to cartesian coordinates

    x_cart = r * np.cos(theta) + scanconversion_config["center_coordinate_pixel"][0]
    y_cart = r * np.sin(theta) + scanconversion_config["center_coordinate_pixel"][1]

    return x_cart, y_cart


def main(args):

    # Make sure output folder exists and save a copy of the configuration file

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Read config file

    if args.scanconvert_config is None:
        args.scanconvert_config = os.path.join(os.path.abspath(os.path.dirname(__file__)), "scanconvert_config.yaml")

    with open(args.scanconvert_config, "r") as f:
        scanconvert_config = yaml.safe_load(f)

    with open(os.path.join(args.output_dir, "scanconvert_config.yaml"), "w") as f:
        yaml.dump(scanconvert_config, f)

    # Print some info on the console

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Config file: {args.scanconvert_config}")
    print(f"Log level: {args.log_level}")
    print(f"Log file: {args.log_file}")

    # Make sure output folder exists and save a copy of the configuration file

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Set up logging into file or console

    if args.log_file is not None:
        log_file = os.path.join(args.output_dir, args.log_file)
        logging.basicConfig(filename=log_file, filemode="w", level=args.log_level)
    else:
        logging.basicConfig(level=args.log_level)  # Log to console

    # Find all data segmentation files and matching ultrasound files in input directory
    
    ultrasound_data_files = sorted(glob.glob(os.path.join(args.input_dir, "*_ultrasound*.npy")))
    segmentation_data_files = sorted(glob.glob(os.path.join(args.input_dir, "*_segmentation*.npy")))
    transform_data_files = sorted(glob.glob(os.path.join(args.input_dir, "*_transform*.npy")))
    
    # Print the number of ultrasound and segmentation files found

    print(f"Found {len(ultrasound_data_files)} ultrasound files.")
    print(f"Found {len(segmentation_data_files)} segmentation files.")
    print(f"Found {len(transform_data_files)} transform files.")

    # Compute cartesian coordinates for inverse scan conversion

    x_cart, y_cart = scan_conversion_inverse(scanconvert_config)

    # Loop over all data files

    for ultrasound_data_file, segmentation_data_file, transform_data_file\
            in tqdm(zip(ultrasound_data_files, segmentation_data_files, transform_data_files), total=len(ultrasound_data_files)):
        
        # Load ultrasound and segmentation data

        ultrasound_data = np.load(ultrasound_data_file)
        segmentation_data = np.load(segmentation_data_file)
        transform_data = np.load(transform_data_file)

        # Create output arrays

        num_lines = scanconvert_config["num_lines"]
        num_samples_along_lines = scanconvert_config["num_samples_along_lines"]

        scanconverted_ultrasound_data = np.zeros((ultrasound_data.shape[0], num_lines, num_samples_along_lines, ultrasound_data.shape[-1]))
        scanconverted_segmentation_data = np.zeros((segmentation_data.shape[0], num_lines, num_samples_along_lines, segmentation_data.shape[-1]))

        # Print some info on the console

        logging.info(f"Loaded {ultrasound_data_file} with shape {ultrasound_data.shape} and value range {np.min(ultrasound_data)} - {np.max(ultrasound_data)}")
        logging.info(f"Loaded {segmentation_data_file} with shape {segmentation_data.shape} and value range {np.min(segmentation_data)} - {np.max(segmentation_data)}")

        # Loop over all frames and all channels in the ultrasound file and perform scan conversion

        for frame_idx in range(ultrasound_data.shape[0]):
            for channel_idx in range(ultrasound_data.shape[-1]):
                scanconverted_ultrasound_data[frame_idx, :, :, channel_idx] =\
                        map_coordinates(ultrasound_data[frame_idx, :, :, channel_idx], [x_cart, y_cart], order=3, mode="nearest")
        
        # Loop over all frames and all channels in the segmentation file and perform scan conversion

        for frame_idx in range(segmentation_data.shape[0]):
            for channel_idx in range(segmentation_data.shape[-1]):
                scanconverted_segmentation_data[frame_idx, :, :, channel_idx] =\
                        map_coordinates(segmentation_data[frame_idx, :, :, channel_idx], [x_cart, y_cart], order=0, mode="nearest")
        
        # Save scan converted data to disk

        scanconverted_ultrasound_data_file = os.path.join(args.output_dir, os.path.basename(ultrasound_data_file))
        scanconverted_segmentation_data_file = os.path.join(args.output_dir, os.path.basename(segmentation_data_file))
        transform_data_file = os.path.join(args.output_dir, os.path.basename(transform_data_file))

        np.save(scanconverted_ultrasound_data_file, scanconverted_ultrasound_data)
        np.save(scanconverted_segmentation_data_file, scanconverted_segmentation_data)
        np.save(transform_data_file, transform_data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
