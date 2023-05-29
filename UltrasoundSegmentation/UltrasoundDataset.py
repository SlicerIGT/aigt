import os
import glob
import numpy as np
from torch.utils.data import Dataset


class UltrasoundDataset(Dataset):
    """
    Dataset class for ultrasound images, segmentations, and transformations.
    """

    def __init__(self, data_folder, transform=None):
        self.transform = transform

        # Find all data segmentation files and matching ultrasound files in input directory
        self.images = sorted(glob.glob(os.path.join(data_folder, "**", "*_ultrasound*.npy"), recursive=True))
        self.segmentations = sorted(glob.glob(os.path.join(data_folder, "**", "*_segmentation*.npy"), recursive=True))
        self.tfm_matrices = sorted(glob.glob(os.path.join(data_folder, "**", "*_transform*.npy"), recursive=True))

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
        Find the datafiles that contains the index and return the image, 
        segmentation, and transform if exists.

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
        ultrasound_data = np.load(self.images[index])
        segmentation_data = np.load(self.segmentations[index])
        transform_data = (np.load(self.tfm_matrices[index]) 
                            if self.tfm_matrices else np.identity(4))
        
        data = {
            "image": ultrasound_data,
            "label": segmentation_data,
            "transform": transform_data
        }

        if self.transform:
            data = self.transform(data)

        return data
