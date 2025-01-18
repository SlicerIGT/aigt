import os
import glob
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

        assert len(self.images) > 0, "No images found in the input directory."
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
    

class GlobalTrackedUSDataset(Dataset):
    def __init__(
            self, 
            root_folder, 
            imgs_dir="images", 
            gts_dir="labels", 
            tfms_dir="transforms", 
            transform=None
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

        # calculate centering translation and scaling for each scan
        print("Calculating centering translation and scaling for each scan...")
        self.norm = {}
        for scan in self.data:
            # load transforms for all frames in one array
            translation = np.stack([
                np.load(self.data[scan]["transform"][i])[:3, 3]
                for i in range(len(self.data[scan]["transform"]))
            ])

            # calculate centering translation matrix
            center = np.mean(translation, axis=0)
            centering_mat = np.eye(4)
            centering_mat[:3, 3] = -center

            # calculate scaling matrix
            min_z = np.min(translation[:, 2])
            max_z = np.max(translation[:, 2])
            range_z = max_z - min_z
            scaling_factor = 2 / range_z
            scaling_mat = np.eye(4)
            scaling_mat[0, 0] = scaling_factor
            scaling_mat[1, 1] = scaling_factor
            scaling_mat[2, 2] = scaling_factor

            # compute final normalization matrix
            norm_mat = scaling_mat @ centering_mat
            self.norm[scan] = norm_mat

    def __len__(self):
        # total number of frames in all scans
        return sum(len(self.data[scan]["image"]) for scan in self.data)

    def __getitem__(self, index):
        scan = None
        for key in self.data:
            scan_len = len(self.data[key]["image"])
            if index < scan_len:
                scan = key
                break
            index -= scan_len
        
        # load data
        image = np.load(self.data[scan]["image"][index])
        label = np.load(self.data[scan]["label"][index])
        transform = np.load(self.data[scan]["transform"][index])

        # If ultrasound_data is 2D, add a channel dimension as last dimension
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        # If segmentation_data is 2D, add a channel dimension as last dimension
        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=-1)
        
        # normalize transformation matrix
        transform = self.norm[scan] @ transform
        transform = np.expand_dims(transform, axis=0)  # add batch dimension
        transform = transform.astype(np.float32)

        data = {
            "image": image,
            "label": label,
            "transform": transform
        }
        
        # apply augmentation
        if self.transform:
            data = self.transform(data)
        
        return data
    

class LocalTrackedUSDataset(Dataset):
    GT_CHANNEL_IDX_FIRST = 0
    GT_CHANNEL_IDX_MIDDLE = 1
    GT_CHANNEL_IDX_LAST = 2

    def __init__(
            self, 
            root_folder, 
            imgs_dir="images", 
            gts_dir="labels", 
            tfms_dir="transforms", 
            transform=None,
            window_size=5,
            gt_idx=GT_CHANNEL_IDX_LAST,
            orig_img_size=512
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

        # which frame to use for ground truth
        if gt_idx == self.GT_CHANNEL_IDX_FIRST:
            self.gt_idx = 0
        elif gt_idx == self.GT_CHANNEL_IDX_MIDDLE:
            self.gt_idx = window_size // 2
        elif gt_idx == self.GT_CHANNEL_IDX_LAST:
            self.gt_idx = window_size - 1
        else:
            raise ValueError("Invalid gt_idx value. Must be 0, 1, or 2.")
        
        # original image size for scaling, can be int or tuple of ints
        if (isinstance(orig_img_size, int) 
            or isinstance(orig_img_size, tuple) and len(orig_img_size) == 1):
            self.img_to_norm = np.diag([*([1 / orig_img_size] * 3), 1])
        elif isinstance(orig_img_size, tuple):
            if len(orig_img_size) == 2:
                l_dim = max(orig_img_size)
                self.img_to_norm = np.diag([
                    1 / orig_img_size[0], 
                    1 / orig_img_size[1], 
                    1 / orig_img_size[l_dim], 
                    1
                ])
            elif len(orig_img_size) == 3:
                self.img_to_norm = np.diag([
                    1 / orig_img_size[0], 
                    1 / orig_img_size[1], 
                    1 / orig_img_size[2], 
                    1
                ])
        else:
            raise ValueError("Invalid orig_img_size. Must be int or tuple.")

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

        # get gt image
        label = np.load(self.data[scan]["label"][index + self.gt_idx])
        # If segmentation_data is 2D, add a channel dimension as last dimension
        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=-1)

        # get ImgToRef transforms
        img_to_ref = np.stack([
            np.load(self.data[scan]["transform"][index + i])
            for i in range(self.window_size)
        ])  # shape: (window_size, 4, 4) - not affected by augmentations

        # calculate ImNToImMain for every other transform and scale
        ref_to_img_main = np.linalg.inv(img_to_ref[self.gt_idx])
        for i in range(self.window_size):
            img_to_ref[i] = self.img_to_norm @ ref_to_img_main @ img_to_ref[i]
        img_to_ref = img_to_ref.astype(np.float32)

        data = {
            "image": image,
            "label": label,
            "transform": img_to_ref
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
    # dataset = LocalTrackedUSDataset("/mnt/e/PerkLab/Data/Spine/SpineTrainingData/04_Slices_train")
    dataset = GlobalTrackedUSDataset("/mnt/c/Users/chris/Data/Spine/2024_SpineSeg/04_Slices_train")
    # dataset = UltrasoundDataset("/mnt/c/Users/chris/Data/Breast/AIGTData/train")
    # print(dataset.images[:5])
    # print(dataset.segmentations[:5])
    # print(dataset.tfm_matrices[:5])
    print(len(dataset))
    print(dataset[0]["image"].shape)
    print(dataset[0]["label"].shape)
    print(dataset[0]["transform"].shape)
    print(dataset[0]["transform"])
