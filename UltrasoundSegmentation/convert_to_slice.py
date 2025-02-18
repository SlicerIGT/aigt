import os
import glob
import argparse
import logging
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--use-file-prefix", action="store_true")
    parser.add_argument("--include-unlabeled-frames", action="store_true")
    parser.add_argument("--log_file", type=str)
    return parser.parse_args()


def main(args):
    # Find all data segmentation files and matching ultrasound files in input directory
    ultrasound_data_files = sorted(glob.glob(os.path.join(args.data_folder, "*_ultrasound*.npy")))
    segmentation_data_files = sorted(glob.glob(os.path.join(args.data_folder, "*_segmentation*.npy")))
    transform_data_files = sorted(glob.glob(os.path.join(args.data_folder, "*_transform*.npy")))
    indices_data_files = sorted(glob.glob(os.path.join(args.data_folder, "*_indices*.npy")))

    # Make sure output folder exists and save a copy of the configuration file

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set up logging into file or console

    if args.log_file is not None:
        log_file = os.path.join(args.output_dir, args.log_file)
        logging.basicConfig(filename=log_file, filemode="w", level="INFO")
    else:
        logging.basicConfig(level="INFO")  # Log to console

    # Create subfolders for images, segmentations, and transforms if they don't exist
    image_dir = os.path.join(args.output_dir, "images")
    label_dir = os.path.join(args.output_dir, "labels")
    tfm_dir = os.path.join(args.output_dir, "transforms")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(tfm_dir, exist_ok=True)

    logging.info(f"Saving individual images, segmentations, and transforms to {args.output_dir}...")

    for pt_idx in tqdm(range(len(ultrasound_data_files))):
        # Create new directory for individual images
        if args.use_file_prefix:
            pt_image_dir = os.path.join(image_dir, os.path.basename(ultrasound_data_files[pt_idx]).rsplit("_", 1)[0])
            pt_label_dir = os.path.join(label_dir, os.path.basename(ultrasound_data_files[pt_idx]).rsplit("_", 1)[0])
            pt_tfm_dir = os.path.join(tfm_dir, os.path.basename(ultrasound_data_files[pt_idx]).rsplit("_", 1)[0])
        else:
            pt_image_dir = os.path.join(image_dir, f"{pt_idx:04d}")
            pt_label_dir = os.path.join(label_dir, f"{pt_idx:04d}")
            pt_tfm_dir = os.path.join(tfm_dir, f"{pt_idx:04d}")
        os.makedirs(pt_image_dir, exist_ok=True)
        os.makedirs(pt_label_dir, exist_ok=True)
        os.makedirs(pt_tfm_dir, exist_ok=True)

        # Read images, segmentations, transforms, and indices
        ultrasound_arr = np.load(ultrasound_data_files[pt_idx])
        segmentation_arr = np.load(segmentation_data_files[pt_idx])
        if transform_data_files:
            transform_arr = np.load(transform_data_files[pt_idx])
        if indices_data_files and indices_data_files[pt_idx]:
            indices_arr = np.load(indices_data_files[pt_idx])
        else:
            indices_arr = None

        seg_idx = 0
        for frame_idx in range(ultrasound_arr.shape[0]):
            # Save individual images
            image_fn = os.path.join(pt_image_dir, f"{frame_idx:04d}_ultrasound.npy")
            np.save(image_fn, ultrasound_arr[frame_idx])

            if indices_arr is not None and args.include_unlabeled_frames:
                if frame_idx in indices_arr:
                    # Save individual segmentations
                    seg_fn = os.path.join(pt_label_dir, f"{frame_idx:04d}_segmentation.npy")
                    np.save(seg_fn, segmentation_arr[seg_idx])
                    seg_idx += 1
            else:
                seg_fn = os.path.join(pt_label_dir, f"{frame_idx:04d}_segmentation.npy")
                np.save(seg_fn, segmentation_arr[frame_idx])

            # Save individual transforms
            if transform_data_files:
                tfm_fn = os.path.join(pt_tfm_dir, f"{frame_idx:04d}_transform.npy")
                np.save(tfm_fn, transform_arr[frame_idx])
    logging.info(f"Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
