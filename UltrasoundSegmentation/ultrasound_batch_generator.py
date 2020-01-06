# Create Ultrasound Segmentation Batch Generator Class

import scipy.ndimage
import numpy as np
import tensorflow as tf


def scale_image(image, factor):
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]

    height, width, depth = image.shape
    zheight = int(np.round(factor * height))
    zwidth = int(np.round(factor * width))
    zdepth = depth

    if factor < 1.0:
        newimg = np.zeros_like(image)
        row = (height - zheight) // 2
        col = (width - zwidth) // 2
        layer = (depth - zdepth) // 2
        newimg[row:row + zheight,
        col:col + zwidth,
        layer:layer + zdepth] = scipy.ndimage.interpolation.zoom(image,
                                                                 (float(factor), float(factor), 1.0),
                                                                 order=0,
                                                                 mode='constant')[0:zheight, 0:zwidth, 0:zdepth]
        return newimg

    elif factor > 1.0:
        row = (zheight - height) // 2
        col = (zwidth - width) // 2
        layer = (zdepth - depth) // 2

        newimg = scipy.ndimage.interpolation.zoom(image[row:row + zheight, col:col + zwidth, layer:layer + zdepth],
                                                  (float(factor), float(factor), 1.0),
                                                  order=0,
                                                  mode='constant')

        extrah = (newimg.shape[0] - height) // 2
        extraw = (newimg.shape[1] - width) // 2
        extrad = (newimg.shape[2] - depth) // 2
        newimg = newimg[extrah:extrah + height, extraw:extraw + width, extrad:extrad + depth]

        return newimg

    else:
        return image


class UltrasoundSegmentationBatchGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 x_set,
                 y_set,
                 batch_size,
                 image_dimensions,
                 shuffle=True,
                 max_rotation_angle = 10.0,
                 max_shift_factor=0.1,
                 min_zoom_factor=0.9,
                 max_zoom_factor=1.1,
                 n_channels=1,
                 n_classes=2):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.image_dimensions = image_dimensions
        self.shuffle = shuffle
        self.max_rotation_angle = max_rotation_angle
        self.max_shift_factor = max_shift_factor
        self.min_zoom_factor = min_zoom_factor
        self.max_zoom_factor = max_zoom_factor
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.number_of_images = self.x.shape[0]
        self.indexes = np.arange(self.number_of_images)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(self.number_of_images / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(self.number_of_images)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        x = np.empty((self.batch_size, *self.image_dimensions, self.n_channels))
        y = np.empty((self.batch_size, *self.image_dimensions))

        for i in range(self.batch_size):
            flip_flag = np.random.randint(2)
            if flip_flag == 1:
                x[i, :, :, :] = np.flip(self.x[batch_indexes[i], :, :, :], axis=1)
                y[i, :, :] = np.flip(self.y[batch_indexes[i], :, :], axis=1)
            else:
                x[i, :, :, :] = self.x[batch_indexes[i], :, :, :]
                y[i, :, :] = self.y[batch_indexes[i], :, :]

        angle = np.random.randint(-self.max_rotation_angle, self.max_rotation_angle)
        x_rot = scipy.ndimage.interpolation.rotate(x, angle, (1, 2), False, mode="constant", cval=0, order=0)
        y_rot = scipy.ndimage.interpolation.rotate(y, angle, (1, 2), False, mode="constant", cval=0, order=0)

        x_shift = np.empty((self.batch_size, *self.image_dimensions, self.n_channels))
        y_shift = np.empty((self.batch_size, *self.image_dimensions))
        for i in range(self.batch_size):
            lower_bound = -int(self.max_shift_factor * self.image_dimensions[0])
            upper_bound = int(self.max_shift_factor * self.image_dimensions[1])
            if lower_bound < upper_bound:
                shift = np.random.randint(lower_bound,  upper_bound, (2))
                x_shift[i, :, :, :] = scipy.ndimage.interpolation.shift(x_rot[i, :, :, :], (shift[0], shift[1], 0),
                                                                        mode="constant", cval=0, order=0)
                y_shift[i, :, :] = scipy.ndimage.interpolation.shift(y_rot[i, :, :], (shift[0], shift[1]),
                                                                     mode="constant",
                                                                     cval=0, order=0)
            else:
                x_shift[i, :, :, :] = x_rot[i, :, :, :]
                y_shift[i, :, :] = y_rot[i, :, :]

        x_zoom = np.empty((self.batch_size, *self.image_dimensions, self.n_channels))
        y_zoom = np.empty((self.batch_size, *self.image_dimensions))
        for i in range(self.batch_size):
            zoom = np.random.uniform(self.min_zoom_factor, self.max_zoom_factor, 1)[0]
            x_zoom[i, :, :, :] = scale_image(x_shift[i, :, :, :], zoom)
            y_zoom[i, :, :] = np.squeeze(scale_image(y_shift[i, :, :], zoom))

        x_out = np.clip(x_zoom, 0.0, 1.0)
        y_out = np.clip(y_zoom, 0.0, 1.0)

        y_onehot = tf.keras.utils.to_categorical(y_out, self.n_classes)
        return x_out, y_onehot
