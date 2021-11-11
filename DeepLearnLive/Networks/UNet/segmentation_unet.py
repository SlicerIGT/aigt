# U-Net Model Construction

import unittest

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import l1, l2

def segmentation_unet_128(input_size, num_classes, num_extra_layers=0, filter_multiplier=10, regularization_rate=0.):
    input_ = Input((input_size, input_size, 1))
    skips = []
    output = input_

    num_layers = int(np.floor(np.log2(input_size)))
    down_conv_kernel_sizes = np.zeros([num_layers], dtype=int)
    down_filter_numbers = np.zeros([num_layers], dtype=int)
    up_conv_kernel_sizes = np.zeros([num_layers], dtype=int)
    up_filter_numbers = np.zeros([num_layers], dtype=int)

    for layer_index in range(num_layers):
        down_conv_kernel_sizes[layer_index] = int(3)
        down_filter_numbers[layer_index] = int((layer_index + 1) * filter_multiplier + num_classes)
        up_conv_kernel_sizes[layer_index] = int(4)
        up_filter_numbers[layer_index] = int((num_layers - layer_index - 1) * filter_multiplier + num_classes)

    for i in range(0, num_extra_layers):
        skips.append(output)
        output = Conv2D(down_filter_numbers[0], kernel_size=3, padding="same", activation="relu", bias_regularizer=l1(regularization_rate))(output)

    for shape, filters in zip(down_conv_kernel_sizes, down_filter_numbers):
        skips.append(output)
        output = Conv2D(filters, (shape, shape), strides=2, padding="same", activation="relu",
                        bias_regularizer=l1(regularization_rate))(output)

    for shape, filters in zip(up_conv_kernel_sizes, up_filter_numbers):
        output = UpSampling2D()(output)
        skip_output = skips.pop()
        output = concatenate([output, skip_output], axis=3)
        if filters != num_classes or input_size == 128:
            output = Conv2D(filters, (shape, shape), activation="relu", padding="same",
                            bias_regularizer=l1(regularization_rate))(output)
            output = BatchNormalization(momentum=.9)(output)
        else:
            output = Conv2D(filters, (shape, shape), activation="softmax", padding="same",
                            bias_regularizer=l1(regularization_rate))(output)

    for i in range(0, num_extra_layers):
        skip_output = skips.pop()
        output = concatenate([output, skip_output], axis=3)
        output = Conv2D(up_filter_numbers[num_layers-1], kernel_size=4, activation="softmax", padding="same", bias_regularizer=l1(regularization_rate))(output)

    assert len(skips) == 0
    return Model([input_], [output])

def segmentation_unet(input_size, num_classes, filter_multiplier=10, regularization_rate=0.):
    input_ = Input((input_size, input_size,1))
    skips = []
    output = input_

    num_layers = int(np.floor(np.log2(input_size)))
    down_conv_kernel_sizes = np.zeros([num_layers], dtype=int)
    down_filter_numbers = np.zeros([num_layers], dtype=int)
    up_conv_kernel_sizes = np.zeros([num_layers], dtype=int)
    up_filter_numbers = np.zeros([num_layers], dtype=int)

    for layer_index in range(num_layers):
        down_conv_kernel_sizes[layer_index] = int(3)
        down_filter_numbers[layer_index] = int((layer_index + 1) * filter_multiplier + num_classes)
        up_conv_kernel_sizes[layer_index] = int(4)
        up_filter_numbers[layer_index] = int((num_layers - layer_index - 1) * filter_multiplier + num_classes)

    for shape, filters in zip(down_conv_kernel_sizes, down_filter_numbers):
        skips.append(output)
        output = Conv2D(filters, (shape, shape), strides=2, padding="same", activation="relu",
                        bias_regularizer=l1(regularization_rate))(output)

    for shape, filters in zip(up_conv_kernel_sizes, up_filter_numbers):
        output = UpSampling2D()(output)
        skip_output = skips.pop()
        output = concatenate([output, skip_output], axis=3)
        if filters != num_classes:
            output = Conv2D(filters, (shape, shape), activation="relu", padding="same",
                            bias_regularizer=l1(regularization_rate))(output)
            output = BatchNormalization(momentum=.9)(output)
        else:
            output = Conv2D(filters, (shape, shape), activation="softmax", padding="same",
                            bias_regularizer=l1(regularization_rate))(output)

    assert len(skips) == 0
    return Model([input_], [output])

def threeChannelUnet(imageShape):
    inputs = Input((imageShape[0], imageShape[1], 3))

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(12, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(12, 1, activation='softmax')(conv9)
    model = Model(input=inputs, output=conv10)

    return model


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    """
    weights = tf.keras.backend.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        loss = y_true * tf.keras.backend.log(y_pred) * weights
        loss = -tf.keras.backend.sum(loss, -1)

        return loss

    return loss


class SagittalSpineUnetTest(unittest.TestCase):
    def test_create_model(self):
        model = segmentation_unet(128, 2)

if __name__ == '__main__':
    unittest.main()