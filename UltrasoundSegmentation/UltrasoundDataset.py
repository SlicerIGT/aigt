import os
import glob
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class UltrasoundDataset(Dataset):
    """
    Dataset class for ultrasound images, segmentations, and transformations.
    Loads data from a directory and keeps one set of datafiles in memory at a time.
    """

    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform

        # Find all data segmentation files and matching ultrasound files in input directory
        ultrasound_data_files = sorted(glob.glob(os.path.join(data_folder, "*_ultrasound*.npy")))
        segmentation_data_files = sorted(glob.glob(os.path.join(data_folder, "*_segmentation*.npy")))
        transform_data_files = sorted(glob.glob(os.path.join(data_folder, "*_transform*.npy")))

        # Save each slice as a separate file temporarily
        self.images = []
        self.segmentations = []
        self.tfm_matrices = []
        logger.info("Saving individual images, segmentations, and transforms to temporary directory...")
        for pt_idx in tqdm(range(len(ultrasound_data_files))):
            # Create new tmp directory for individual images
            pt_tmp_dir = os.path.join(data_folder, "tmp", f"{str(pt_idx):04d}")
            os.makedirs(pt_tmp_dir, exist_ok=True)

            # Read images, segmentations, and transforms
            ultrasound_arr = np.load(ultrasound_data_files[pt_idx])
            segmentation_arr = np.load(segmentation_data_files[pt_idx])
            if transform_data_files:
                transform_arr = np.load(transform_data_files[pt_idx])

            for frame_idx in range(ultrasound_arr.shape[0]):
                # Save individual images
                image_fn = os.path.join(pt_tmp_dir, f"{str(frame_idx):04d}_ultrasound.npy")
                np.save(image_fn, ultrasound_arr[frame_idx])
                self.images.append(image_fn)

                # Save individual segmentations
                seg_fn = os.path.join(pt_tmp_dir, f"{str(frame_idx):04d}_segmentation.npy")
                np.save(seg_fn, segmentation_arr[frame_idx])
                self.segmentations.append(seg_fn)

                # Save individual transforms
                if transform_data_files:
                    tfm_fn = os.path.join(pt_tmp_dir, f"{str(frame_idx):04d}_transform.npy")
                    np.save(tfm_fn, transform_arr[frame_idx])
                    self.tfm_matrices.append(tfm_fn)
        logger.info(f"Done. Data stored in {pt_tmp_dir}.")

    def __len__(self):
        """
        Returns the total number of segmented images in the dataset.
        
        Returns
        -------
        int
            Total number of segmented images in the dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Find the datafiles that contains the index and return the image, segmentation, and transform if exists.
        Keep one set of datafiles open and only load another set of datafiles when the requested index is not in the already loaded datafile.

        Parameters
        ----------
        index : int
            Index of the image, segmentation, and transform to return

        Returns
        -------
        image : numpy array
            Ultrasound image
        segmentation : numpy array
            Segmentation of the ultrasound image
        transform : numpy array
            Transform of the ultrasound image
        """
        # Read the image and segmentation from temp directory
        ultrasound_data = np.load(self.images[index])
        segmentation_data = np.load(self.segmentations[index])

        # If segmentation_data only has 3 dimensions, expand it
        if segmentation_data.ndim == 3:
            np.expand_dims(segmentation_data, -1)
        
        data = {
            "image": ultrasound_data,
            "label": segmentation_data,
            "transform": (np.load(self.transform_data[index]) 
                          if self.transform_data_files else np.identity(4))
        }

        if self.transform is not None:
            data = self.transform(data)

        return data
