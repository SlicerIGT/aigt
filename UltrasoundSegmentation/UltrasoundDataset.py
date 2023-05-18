import os
import glob
import atexit
import shutil
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

    def __init__(self, data_folder, transform=None, shuffle=False):
        self.data_folder = data_folder
        self.transform = transform
        self.shuffle = shuffle
        tmp_folder = os.path.join(data_folder, "tmp")

        # Find all data segmentation files and matching ultrasound files in input directory
        ultrasound_data_files = sorted(glob.glob(os.path.join(data_folder, "*_ultrasound*.npy")))
        segmentation_data_files = sorted(glob.glob(os.path.join(data_folder, "*_segmentation*.npy")))
        transform_data_files = sorted(glob.glob(os.path.join(data_folder, "*_transform*.npy")))

        if shuffle:
            # Save each slice as a separate file temporarily
            self.images = []
            self.segmentations = []
            self.tfm_matrices = []
            logger.info(f"Saving individual images, segmentations, and transforms to {tmp_folder}...")
            for pt_idx in tqdm(range(len(ultrasound_data_files))):
                # Create new tmp directory for individual images
                pt_tmp_dir = os.path.join(tmp_folder, f"{pt_idx:04d}")
                os.makedirs(pt_tmp_dir, exist_ok=True)

                # Read images, segmentations, and transforms
                ultrasound_arr = np.load(ultrasound_data_files[pt_idx])
                segmentation_arr = np.load(segmentation_data_files[pt_idx])
                if transform_data_files:
                    transform_arr = np.load(transform_data_files[pt_idx])

                for frame_idx in range(ultrasound_arr.shape[0]):
                    # Save individual images
                    image_fn = os.path.join(pt_tmp_dir, f"{frame_idx:04d}_ultrasound.npy")
                    np.save(image_fn, ultrasound_arr[frame_idx])
                    self.images.append(image_fn)

                    # Save individual segmentations
                    seg_fn = os.path.join(pt_tmp_dir, f"{frame_idx:04d}_segmentation.npy")
                    np.save(seg_fn, segmentation_arr[frame_idx])
                    self.segmentations.append(seg_fn)

                    # Save individual transforms
                    if transform_data_files:
                        tfm_fn = os.path.join(pt_tmp_dir, f"{frame_idx:04d}_transform.npy")
                        np.save(tfm_fn, transform_arr[frame_idx])
                        self.tfm_matrices.append(tfm_fn)
            logger.info(f"Done.")
        else:  
            # Load images sequentially into memory to speed up training
            self.images = ultrasound_data_files
            self.segmentations = segmentation_data_files
            self.tfm_matrices = transform_data_files

            # Save the lengths of each data file so we can find the correct file when we need to load data
            self.data_file_lengths = [np.load(data_file).shape[0] for data_file in segmentation_data_files]

            # Load the first datafile into memory
            self.data_file_index = 0
            self.segmentation_data = np.load(segmentation_data_files[self.data_file_index])
            self.ultrasound_data = np.load(ultrasound_data_files[self.data_file_index])
            if transform_data_files:
                self.transform_data = np.load(transform_data_files[self.data_file_index])

        def cleanup():
            """
            Remove temporary directory with individual images, segmentations, and transforms.
            """
            if os.path.exists(tmp_folder):
                logger.info(f"Removing temporary directory {tmp_folder}...")
                shutil.rmtree(tmp_folder)
                logger.info(f"Done.")
            
        atexit.register(cleanup)

    def __len__(self):
        """
        Returns the total number of segmented images in the dataset.
        
        Returns
        -------
        int
            Total number of segmented images in the dataset
        """
        if self.shuffle:
            dataset_len = len(self.images)
        else:
            dataset_len = sum(self.data_file_lengths)
        return dataset_len

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
        if self.shuffle:
            # Read the image and segmentation from temp directory
            ultrasound_data = np.load(self.images[index])
            segmentation_data = np.load(self.segmentations[index])
            transform_data = (np.load(self.tfm_matrices[index]) 
                              if self.tfm_matrices else np.identity(4))
        else:
            # Find the datafile that contains the index
            data_file_index = 0
            while index >= self.data_file_lengths[data_file_index]:
                index -= self.data_file_lengths[data_file_index]
                data_file_index += 1

            # Load the datafile if it is not already loaded
            if data_file_index != self.data_file_index:
                self.segmentation_data = np.load(self.segmentations[data_file_index])
                self.ultrasound_data = np.load(self.images[data_file_index])
                if self.tfm_matrices:
                    self.transform_data = np.load(self.tfm_matrices[data_file_index])
                self.data_file_index = data_file_index

            # Return the image, segmentation, and transform if exists
            ultrasound_data = self.ultrasound_data[index]
            segmentation_data = self.segmentation_data[index]
            transform_data = (self.transform_data[index] 
                              if self.tfm_matrices else np.identity(4))
        
        data = {
            "image": ultrasound_data,
            "label": segmentation_data,
            "transform": transform_data
        }

        if self.transform is not None:
            data = self.transform(data)

        return data
