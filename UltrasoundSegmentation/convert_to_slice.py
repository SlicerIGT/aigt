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
    return parser.parse_args()


def main(args):
    # Find all data segmentation files and matching ultrasound files in input directory
    ultrasound_data_files = sorted(glob.glob(os.path.join(args.data_folder, "*_ultrasound*.npy")))
    segmentation_data_files = sorted(glob.glob(os.path.join(args.data_folder, "*_segmentation*.npy")))
    transform_data_files = sorted(glob.glob(os.path.join(args.data_folder, "*_transform*.npy")))

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
            pt_image_dir = os.path.join(image_dir, os.path.basename(ultrasound_data_files[pt_idx]).split("_")[0])
            pt_label_dir = os.path.join(label_dir, os.path.basename(ultrasound_data_files[pt_idx]).split("_")[0])
            pt_tfm_dir = os.path.join(tfm_dir, os.path.basename(ultrasound_data_files[pt_idx]).split("_")[0])
        else:
            pt_image_dir = os.path.join(image_dir, f"{pt_idx:04d}")
            pt_label_dir = os.path.join(label_dir, f"{pt_idx:04d}")
            pt_tfm_dir = os.path.join(tfm_dir, f"{pt_idx:04d}")
        os.makedirs(pt_image_dir, exist_ok=True)
        os.makedirs(pt_label_dir, exist_ok=True)
        os.makedirs(pt_tfm_dir, exist_ok=True)

        # Read images, segmentations, and transforms
        ultrasound_arr = np.load(ultrasound_data_files[pt_idx])
        segmentation_arr = np.load(segmentation_data_files[pt_idx])
        if transform_data_files:
            transform_arr = np.load(transform_data_files[pt_idx])

        for frame_idx in range(ultrasound_arr.shape[0]):
            # Save individual images
            image_fn = os.path.join(pt_image_dir, f"{frame_idx:04d}_ultrasound.npy")
            np.save(image_fn, ultrasound_arr[frame_idx])

            # Save individual segmentations
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
