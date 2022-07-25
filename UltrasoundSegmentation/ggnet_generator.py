# Create Ultrasound Segmentation Batch Generator Class

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input


class UltrasoundSegmentationBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 x_set,
                 y_set,
                 batch_size,
                 image_dimensions,
                 transforms=None,
                 shuffle=True,
                 n_channels=1,
                 n_classes=2,
                 rng=np.random.default_rng()):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.image_dimensions = image_dimensions
        self.transforms = transforms
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.rng = rng
        self.number_of_images = self.x.shape[0]
        self.indexes = np.arange(self.number_of_images)
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def input_parser(self, img, label):
        sample = {"image": img, "label": label}

        # Apply data augmentation
        with tf.device("cpu:0"):
            if self.transforms:
                for transform in self.transforms:
                    sample = transform(sample, self.rng)

        return sample["image"], sample["label"]

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __len__(self):
        return self.number_of_images // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x_batch = np.zeros((0, *self.image_dimensions, self.n_channels))
        y_batch = np.zeros((0, *self.image_dimensions, 1))

        # Augment images then add to batch
        for i in range(self.batch_size):
            img, label = self.input_parser(self.x[batch_indexes[i]], self.y[batch_indexes[i]])
            x_batch = np.concatenate((x_batch, np.expand_dims(img, axis=0)))
            y_batch = np.concatenate((y_batch, np.expand_dims(label, axis=0)))

        x_batch = np.clip(x_batch, 0.0, 1.0) * 255
        y_batch = np.clip(y_batch, 0.0, 1.0)

        # Convert to 3-channel and process for resnet
        x_batch = tf.convert_to_tensor(x_batch)
        x_batch = tf.repeat(x_batch, repeats=3, axis=-1)
        x_batch = preprocess_input(x_batch)

        # Generate boundary ground truth
        y_batch_boundary = np.zeros((0, *self.image_dimensions, 1))
        for i in range(self.batch_size):
            y_boundary = (y_batch[i, ..., 0] * 255).astype(np.uint8)
            y_boundary = cv2.Canny(y_boundary, 100, 200).astype(np.float32)
            y_boundary /= 255.
            y_batch_boundary = np.concatenate((y_batch_boundary, y_boundary[np.newaxis, ..., np.newaxis]))
        y_batch_boundary = tf.convert_to_tensor(y_batch_boundary, dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)

        return x_batch, y_batch, y_batch_boundary


class RandomScale:
    """Randomly scales an image by a scaling factor."""
    def __init__(self, scaling_limit=0.1):
        self.name = "RandomScale"
        self.scaling_limit = scaling_limit

    def __call__(self, sample, rng=np.random.default_rng()):
        # Randomly determine scaling factor
        scaling_factor = rng.uniform(1 - self.scaling_limit, 1 + self.scaling_limit)
        interpolation = cv2.INTER_AREA if scaling_factor < 1 else cv2.INTER_LINEAR

        # Generate transformation matrix
        img, label = sample["image"], sample["label"]
        height, width = img.shape[:2]
        box0 = np.array(
            [[0, 0],
             [width, 0],
             [width, height],
             [0, height]]
        )
        box1 = box0 * scaling_factor

        # Apply transformation to image and label
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        img = np.expand_dims(cv2.warpPerspective(img, mat, (width, height), flags=interpolation), axis=2)
        label = np.expand_dims(cv2.warpPerspective(label, mat, (width, height), flags=interpolation), axis=2)

        return {"image": img, "label": label}


class RandomShift:
    """Randomly shifts an image by a proportion of the image dimensions."""
    def __init__(self, shift_factor=0.1):
        self.name = "RandomShift"
        self.shift_factor = shift_factor

    def __call__(self, sample, rng=np.random.default_rng()):
        # Randomly get distance to shift image in x and y
        img, label = sample["image"], sample["label"]
        height, width = img.shape[:2]
        max_dx = self.shift_factor * width
        max_dy = self.shift_factor * height
        dx = round(rng.uniform(-max_dx, max_dx))
        dy = round(rng.uniform(-max_dy, max_dy))

        # Generate transformation matrix
        box0 = np.array(
            [[0, 0],
             [width, 0],
             [width, height],
             [0, height]]
        )
        box1 = box0 - np.array([width / 2, height / 2])
        box1 += np.array([width / 2 + dx, height / 2 + dy])

        # Apply transformation to image and label
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        img = np.expand_dims(cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR), axis=2)
        label = np.expand_dims(cv2.warpPerspective(label, mat, (width, height), flags=cv2.INTER_LINEAR), axis=2)

        return {"image": img, "label": label}


class RandomFlip:
    """Randomly flips an image horizontally and vertically."""
    def __init__(self, p=0.5):
        self.name = "RandomFlip"
        self.p = p

    def __call__(self, sample, rng=np.random.default_rng()):
        img, label = sample["image"], sample["label"]

        # Horizontal flip
        if rng.random() < self.p:
            seed = rng.integers(np.iinfo(np.int64).max, size=(2, ))
            img = tf.image.stateless_random_flip_left_right(img, seed)
            label = tf.image.stateless_random_flip_left_right(label, seed)

        # Vertical flip
        if rng.random() < self.p:
            seed = rng.integers(np.iinfo(np.int64).max, size=(2,))
            img = tf.image.stateless_random_flip_up_down(img, seed)
            label = tf.image.stateless_random_flip_up_down(label, seed)

        return {"image": img, "label": label}


class RandomRotation:
    """Randomly rotates an image 0, 90, 180, or 270 degrees."""
    def __init__(self):
        self.name = "RandomRotation"

    def __call__(self, sample, rng=np.random.default_rng()):
        img, label = sample["image"], sample["label"]

        num_rot = rng.integers(4)  # Number of 90 degree rotations
        img = tf.image.rot90(img, num_rot)
        label = tf.image.rot90(label, num_rot)

        return {"image": img, "label": label}
