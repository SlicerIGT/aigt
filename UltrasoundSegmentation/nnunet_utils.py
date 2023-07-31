import os
import glob
import json
import logging
import numpy as np
import nibabel as nib
import torch

from nnUNet.nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnUNet.nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity

logger = logging.getLogger(__name__)


def convert_to_nnunet_raw(train_dir, val_dir, nnunet_raw_dir, dataset_name, 
                          channel_names, labels, verify_dataset=False):
    dataset_dir = os.path.join(nnunet_raw_dir, dataset_name)
    image_tr = os.path.join(dataset_dir, "imagesTr")
    label_tr = os.path.join(dataset_dir, "labelsTr")

    # Create nnUnet raw data folders
    if not (os.path.isfile(os.path.join(dataset_dir, "dataset.json"))
            and os.path.isdir(image_tr)
            and os.path.isdir(label_tr)):

        logger.info("Converting dataset to nnUNet data format...")
        os.makedirs(image_tr, exist_ok=True)
        os.makedirs(label_tr, exist_ok=True)

        # Get names of all training images
        tr_image_list = glob.glob(os.path.join(train_dir, "**/*_ultrasound*.npy"), recursive=True)
        tr_label_list = glob.glob(os.path.join(train_dir, "**/*_segmentation*.npy"), recursive=True)
        val_image_list = glob.glob(os.path.join(val_dir, "**/*_ultrasound*.npy"), recursive=True)
        val_label_list = glob.glob(os.path.join(val_dir, "**/*_segmentation*.npy"), recursive=True)
        tr_images = tr_image_list + val_image_list
        tr_labels = tr_label_list + val_label_list
        tr_images = [os.path.normpath(image) for image in tr_images]
        tr_labels = [os.path.normpath(label) for label in tr_labels]
        
        assert len(tr_images) == len(tr_labels), "Number of training images and labels do not match"

        # Copy training data to nnUnet raw data folders
        for image, label in zip(tr_images, tr_labels):
            # Load images and convert to 2D nifti
            img_arr = np.load(image).astype(np.float64).transpose(1, 0, 2)
            label_arr = np.load(label).astype(np.float64).transpose(1, 0, 2)
            img_nifti = nib.Nifti1Image(img_arr, np.eye(4))
            label_nifti = nib.Nifti1Image(label_arr, np.eye(4))

            # Save nifti images to nnUNet directory
            case_id = image.split(os.sep)[-2] + "_" + os.path.basename(image).split("_")[0]
            nib.save(img_nifti, os.path.join(image_tr, case_id + "_0000.nii.gz"))
            nib.save(label_nifti, os.path.join(label_tr, case_id + ".nii.gz"))
        
        # Create dataset.json
        generate_dataset_json(
            dataset_dir,
            channel_names=channel_names,
            labels=labels,
            num_training_cases=len(tr_images),
            file_ending=".nii.gz"
        )
        logger.info(f"Done. nnUNet dataset created at {dataset_dir}.")
    else:
        logger.info(f"found nnUNet dataset at {dataset_dir}.")
    
    # Sanity check
    if verify_dataset:
        verify_dataset_integrity(dataset_dir)


def generate_split_json(train_dir, val_dir, preprocessed_dataset_dir):
    # Get paths of all training and validation images
    tr_image_list = glob.glob(os.path.join(train_dir, "**/*_ultrasound*.npy"), recursive=True)
    val_image_list = glob.glob(os.path.join(val_dir, "**/*_ultrasound*.npy"), recursive=True)
    tr_image_list = [os.path.normpath(image) for image in tr_image_list]
    val_image_list = [os.path.normpath(image) for image in val_image_list]

    # Get case IDs
    tr_ids = [image.split(os.sep)[-2] + "_" + os.path.basename(image).split("_")[0]
              for image in tr_image_list]
    val_ids = [image.split(os.sep)[-2] + "_" + os.path.basename(image).split("_")[0]
               for image in val_image_list]
    splits = [{
        "train": tr_ids, 
         "val": val_ids
    }]

    # Save as json to nnUNet_preprocessed (will be loaded automatically by nnUNet)
    with open(os.path.join(preprocessed_dataset_dir, "splits_final.json") , 'w') as f:
        json.dump(splits, f)
    logger.info(f"splits_final.json created at {preprocessed_dataset_dir}")


if __name__ == "__main__":
    train_dir = "d:/UltrasoundSegmentation/Breast/data/train"
    val_dir = "d:/UltrasoundSegmentation/Breast/data/val"
    nnunet_raw_dir = "d:/UltrasoundSegmentation/Breast/data/nnUNet_raw"
    dataset_name = "Dataset001_Breast"
    convert_to_nnunet_raw(train_dir, val_dir, nnunet_raw_dir, dataset_name)
    # generate_split_json(train_dir, val_dir, os.path.join(nnunet_raw_dir, dataset_name))
