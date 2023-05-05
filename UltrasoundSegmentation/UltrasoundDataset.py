import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class UltrasoundDataset(Dataset):
    """
    Dataset class for ultrasound images, segmentations, and transformations.
    Loads data from a directory and keeps one set of datafiles in memory at a time.
    """

    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform

        # Find all data segmentation files and matching ultrasound files in input directory
        segmentation_data_files = []
        ultrasound_data_files = []
        transform_data_files = []
        for filename in os.listdir(data_folder):
            if filename.endswith(".npy") and "_segmentation" in filename:
                segmentation_data_files.append(os.path.join(data_folder, filename))
                ultrasound_data_files.append(os.path.join(data_folder, filename.replace("_segmentation", "_ultrasound")))
                transform_data_files.append(os.path.join(data_folder, filename.replace("_segmentation", "_transform")))
        
        # Check if all trasnform files exist, and if not, disable transforms
        if not all(os.path.exists(transform_file) for transform_file in transform_data_files):
            transform_data_files = None

        self.segmentation_data_files = segmentation_data_files
        self.ultrasound_data_files = ultrasound_data_files
        self.transform_data_files = transform_data_files

        # Save the lengths of each data file so we can find the correct file when we need to load data
        self.data_file_lengths = [np.load(data_file).shape[0] for data_file in self.segmentation_data_files]

        # Load the first datafile into memory
        self.data_file_index = 0
        self.segmentation_data = np.load(self.segmentation_data_files[self.data_file_index])
        self.ultrasound_data = np.load(self.ultrasound_data_files[self.data_file_index])
        if self.transform_data_files is not None:
            self.transform_data = np.load(self.transform_data_files[self.data_file_index])


    def __len__(self):
        """
        Returns the total number of segmented images in the dataset.
        
        Returns
        -------
        int
            Total number of segmented images in the dataset
        """
        return sum(self.data_file_lengths)
    

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

        # Find the datafile that contains the index
        data_file_index = 0
        while index >= self.data_file_lengths[data_file_index]:
            index -= self.data_file_lengths[data_file_index]
            data_file_index += 1

        # Load the datafile if it is not already loaded
        if data_file_index != self.data_file_index:
            self.segmentation_data = np.load(self.segmentation_data_files[data_file_index])
            self.ultrasound_data = np.load(self.ultrasound_data_files[data_file_index])
            if self.transform_data_files is not None:
                self.transform_data = np.load(self.transform_data_files[data_file_index])
            self.data_file_index = data_file_index

        # Return the image, segmentation, and transform if exists
        ultrasound_data = self.ultrasound_data[index]
        segmentation_data = self.segmentation_data[index]

        # If segmentation_data only has 3 dimensions, expand it
        if segmentation_data.ndim == 3:
            np.expand_dims(segmentation_data, -1)

        if self.transform is not None:
            ultrasound_data = self.transform(torch.from_numpy(np.transpose(ultrasound_data, (2,0,1)))).float()
            segmentation_data = self.transform(torch.from_numpy(np.transpose(segmentation_data, (2,0,1)))).float()

        if self.transform_data_files is not None:
            return ultrasound_data, segmentation_data, self.transform_data[index]
        else:
            return ultrasound_data, segmentation_data, None        
