import torch
import torch.nn as nn


class TrackingModule(nn.Module):
    def __init__(self, image_size, window_size):
        super().__init__()
        self.window_size = window_size
        self.image_size = image_size

        num_pixels = image_size * image_size
        self.fc1 = nn.Linear(16, num_pixels)
        self.fc2 = nn.Linear(num_pixels, num_pixels)
        
    def forward(self, inputs):
        x, tfms = inputs
        batch_size, _, _, _ = x.shape

        tfms = tfms.view(batch_size, self.window_size, -1)  # (batch_size, window_size, 16)
        tfms = self.fc1(tfms)
        tfms = self.fc2(tfms)
        tfms = tfms.view(batch_size, self.window_size, self.image_size, self.image_size)

        # add the transformation with the input image
        x = x + tfms
        return x


if __name__ == "__main__":
    import datasets
    import monai
    from torch.utils.data import DataLoader
    from monai.transforms import (
        Compose,
        Transposed,
        ToTensord,
        EnsureTyped
    )

    train_dir = "/mnt/c/Users/chris/Data/Spine/2024_SpineSeg/04_Slices_train"

    transforms = Compose([
        Transposed(keys=["image", "label"], indices=[2, 0, 1]),
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32)
    ])

    # dataset = datasets.LocalTrackedUSDataset(train_dir, transform=transforms)
    dataset = datasets.GlobalTrackedUSDataset(train_dir, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))
    inputs = batch["image"]
    tfms = batch["transform"]
    print(f"inputs: {inputs.shape}, tfms: {tfms.shape}")

    unet = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )
    tracking = TrackingModule(128, 1)
    model = nn.Sequential(tracking, unet)
    output = model((inputs, tfms))
    print(f"output: {output.shape}")
