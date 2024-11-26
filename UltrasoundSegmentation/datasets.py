import os
import glob
import vtk
import numpy as np
from torch.utils.data import Dataset
from monai.config import KeysCollection
from monai.transforms.transform import MapTransform


class UltrasoundDataset(Dataset):
    """
    Dataset class for ultrasound images, segmentations, and transformations.
    """

    def __init__(self, root_folder, imgs_dir="images", gts_dir="labels", tfms_dir="transforms", transform=None):
        self.transform = transform

        # Find all data segmentation files and matching ultrasound files in input directory
        self.images = sorted(glob.glob(os.path.join(root_folder, "**", imgs_dir, "**", "*.npy"), recursive=True))
        self.segmentations = sorted(glob.glob(os.path.join(root_folder, "**", gts_dir, "**", "*.npy"), recursive=True))
        self.tfm_matrices = sorted(glob.glob(os.path.join(root_folder, "**", tfms_dir, "**", "*.npy"), recursive=True))
        assert len(self.images) == len(self.segmentations), "Number of images and segmentations must match."

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
        
        # If ultrasound_data is 2D, add a channel dimension as last dimension
        if len(ultrasound_data.shape) == 2:
            ultrasound_data = np.expand_dims(ultrasound_data, axis=-1)
            
        # If segmentation_data is 2D, add a channel dimension as last dimension
        if len(segmentation_data.shape) == 2:
            segmentation_data = np.expand_dims(segmentation_data, axis=-1)

        if len(transform_data.shape) == 2:
            transform_data = np.expand_dims(transform_data, axis=0)
        
        data = {
            "image": ultrasound_data,
            "label": segmentation_data,
            "transform": transform_data
        }

        if self.transform:
            data = self.transform(data)

        return data
    

class SlidingWindowTrackedUSDataset(Dataset):
    def __init__(
            self, 
            root_folder, 
            imgs_dir="images", 
            gts_dir="labels", 
            tfms_dir="transforms", 
            transform=None,
            window_size=4
        ):
        # get names of subfolders in imgs_dir, gts_dir, and tfms_dir
        image_scans = [
            f.name for f in os.scandir(
                os.path.join(root_folder, imgs_dir)
            ) if f.is_dir()
        ]
        gt_scans = [
            f.name for f in os.scandir(
                os.path.join(root_folder, gts_dir)
            ) if f.is_dir()
        ]
        tfm_scans = [
            f.name for f in os.scandir(
                os.path.join(root_folder, tfms_dir)
            ) if f.is_dir()
        ]
        assert set(image_scans) == set(gt_scans) == set(tfm_scans), \
            "Scans in images, labels, and transforms directories must be the same."
        
        # get file paths for each scan
        self.data = {
            scan: {
                "image": sorted(glob.glob(os.path.join(
                    root_folder, imgs_dir, scan, "*.npy"
                ))),
                "label": sorted(glob.glob(os.path.join(
                    root_folder, gts_dir, scan, "*.npy"
                ))),
                "transform": sorted(glob.glob(os.path.join(
                    root_folder, tfms_dir, scan, "*.npy"
                )))
            } for scan in image_scans
        }

        self.transform = transform
        self.window_size = window_size

    def __len__(self):
        return sum(
            len(self.data[scan]["image"]) - self.window_size + 1
            for scan in self.data
        )
    
    def __getitem__(self, index):
        scan = None
        for key in self.data:
            scan_len = len(self.data[key]["image"]) - self.window_size + 1
            if index < scan_len:
                scan = key
                break
            index -= scan_len
        
        image = np.stack([
            np.load(self.data[scan]["image"][index + i])[..., 0]
            for i in range(self.window_size)
        ], axis=-1)  # shape: (H, W, window_size)

        # only take middle frame as label
        label = np.load(self.data[scan]["label"][index + self.window_size // 2])
        # If segmentation_data is 2D, add a channel dimension as last dimension
        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=-1)

        transform = np.stack([
            np.load(self.data[scan]["transform"][index + i])
            for i in range(self.window_size)
        ])  # shape: (window_size, 4, 4) - not affected by transforms

        # define 3 points based on max x, y, and z coordinates of set of transforms
        from_points = vtk.vtkPoints()
        from_points.SetNumberOfPoints(3)
        from_points.SetPoint(0, np.max(transform[:, 0, 3]), 0, 0)
        from_points.SetPoint(1, 0, np.max(transform[:, 1, 3]), 0)
        from_points.SetPoint(2, 0, 0, np.max(transform[:, 2, 3]))

        to_points = vtk.vtkPoints()
        to_points.SetNumberOfPoints(3)
        to_points.SetPoint(0, 1, 0, 0)
        to_points.SetPoint(1, 0, 1, 0)
        to_points.SetPoint(2, 0, 0, 1)

        # fiducial registration
        landmarkTransform = vtk.vtkLandmarkTransform()
        landmarkTransform.SetSourceLandmarks(from_points)
        landmarkTransform.SetTargetLandmarks(to_points)
        landmarkTransform.SetModeToSimilarity()
        landmarkTransform.Update()

        # get the transformation matrix
        matrix = vtk.vtkMatrix4x4()
        landmarkTransform.GetMatrix(matrix)
        img_to_norm = np.eye(4)
        matrix.DeepCopy(img_to_norm.ravel(), matrix)

        # apply transformation to each frame
        for i in range(self.window_size):
            transform[i] = img_to_norm @ transform[i]
        
        data = {
            "image": image,
            "label": label,
            "transform": transform
        }
        
        if self.transform:
            data = self.transform(data)
        
        return data
    

class ZScoreNormalized(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = (d[key] - np.mean(d[key])) / max(np.std(d[key]), 1e-8)
        return d
    

if __name__ == "__main__":
    dataset = SlidingWindowTrackedUSDataset("/mnt/e/PerkLab/Data/Spine/SpineTrainingData/04_Slices_train")
    # dataset = UltrasoundDataset("/mnt/c/Users/chris/Data/Breast/AIGTData/train")
    # print(dataset.images[:5])
    # print(dataset.segmentations[:5])
    # print(dataset.tfm_matrices[:5])
    print(len(dataset))
    print(dataset[0]["image"].shape)
    print(dataset[0]["label"].shape)
    print(dataset[0]["transform"].shape)
    print(dataset[0]["transform"])
