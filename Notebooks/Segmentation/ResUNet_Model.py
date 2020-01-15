import os
import random
import time
import numpy as np
from math import ceil
import logging


import cv2
from PIL import Image
from PIL import ImageFilter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, Input, BatchNormalization, UpSampling2D
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Model, load_model
#from tensorflow.keras.callbacks import EarlyStopping


Feature_layer1=10      # Number of feature maps in layer 1  
Feature_layer2=18     # Number of feature maps in layer 2
Feature_layer3=26     # Number of feature maps in layer 3
Feature_layer4=34     # Number of feature maps in layer 4
Feature_layer5=42     # Number of feature maps in layer 5
Feature_layer6=50      # Number of feature maps in layer 6

dialated_conv1 = 1
dialated_conv2 = 2
dialated_conv3 = 4

weightfactor = 3 # weightfactor for the amplitude where you want to provide more weight. Recommend to use a small validation set to fix this hyper-parameter


def _resnet_layer(bottom, filters, dilationrate, regularization_rate=.0001):
    #Left side of the resnet layer, lower level features
    x0 = Conv2D(filters,1,padding='SAME', bias_regularizer=l1(regularization_rate))(bottom)
    
    
    #Right side of the resnet layer, higher level features
    x = Conv2D(filters,1,padding='SAME', bias_regularizer=l1(regularization_rate))(bottom)
    x = BatchNormalization(momentum=.9)(x)
    x = keras.layers.Activation("relu")(x)

    x = Conv2D(filters,3,padding='SAME',dilation_rate=dilationrate, bias_regularizer=l1(regularization_rate))(x)
    x = BatchNormalization(momentum=.9)(x)
    keras.layers.Activation("relu")(x)

    x = Conv2D(filters,1,padding='SAME', bias_regularizer=l1(regularization_rate))(x)
    x = BatchNormalization(momentum=.9)(x)

    #combine higher level and lower level features
    return tf.nn.relu(keras.layers.Add()([x,x0]))
    
def _upscore_layer(bottom, ksize=2, stride=2):
    in_features = bottom.shape[3]
    f_shape = [ksize, ksize, in_features, in_features]
    weights = get_deconv_filter(f_shape)
    deconv = Conv2DTranspose(in_features, kernel_size=ksize, strides=stride, kernel_initializer=weights)(bottom)
    return deconv
  
def get_deconv_filter(f_shape):
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = Constant(value=weights)
    return init
      
def _feature_extraction(bottom):
    #This is the expanding branch in figure 2a of Emran's paper
    
    #Layer1_1E + Layer1_2E + pool_1E
    conv1_1 = _resnet_layer(bottom, Feature_layer1, 1)
    conv1_2 = _resnet_layer(conv1_1, Feature_layer1, 1)
    pool1 = MaxPool2D(pool_size=2, strides=2)(conv1_2)
    
    #Layer2_1E + Layer2_2E + pool_2E
    conv2_1 = _resnet_layer(pool1, Feature_layer2, 1)
    conv2_2 = _resnet_layer(conv2_1, Feature_layer2, 1)
    pool2 = MaxPool2D(pool_size=2, strides=2)(conv2_2)
    
    #Layer3_1E + Layer3_2E + pool_3E
    conv3_1 = _resnet_layer(pool2, Feature_layer3, 1)
    conv3_2 = _resnet_layer(conv3_1, Feature_layer3, 1)
    pool3 = MaxPool2D(pool_size=2, strides=2)(conv3_2)
    
    #no more pooling
    #used dialated convolution
    #dialated convolution there is spacing and we expand size of kernel
    
    #Layer4_1E + Layer4_2E, dialation = 1
    conv4_1 = _resnet_layer(pool3, Feature_layer4, dialated_conv1)
    conv4_2 = _resnet_layer(conv4_1, Feature_layer4, dialated_conv1)
    
    #Layer5_1E + Layer5_2E, dialation = 2 meaning we skip 2
    conv5_1 = _resnet_layer(conv4_2, Feature_layer5, dialated_conv2)
    conv5_2 = _resnet_layer(conv5_1, Feature_layer5, dialated_conv2)
    
    #Layer6_1E + Layer6_2E, dialation = 4 meaning skip between two kernels is 3
    conv6_1 = _resnet_layer(conv5_2, Feature_layer6, dialated_conv3)
    conv6_2 = _resnet_layer(conv6_1, Feature_layer6, dialated_conv3)
    
    return conv6_2
    
def _interpolation(bottom):
    conv6_1R = _resnet_layer(bottom, Feature_layer6, dialated_conv3)
    conv6_2R = _resnet_layer(conv6_1R, Feature_layer6,  dialated_conv3)

    conv5_1R =_resnet_layer(conv6_2R, Feature_layer5, dialated_conv2)
    conv5_2R = _resnet_layer(conv5_1R, Feature_layer5, dialated_conv2)

    conv4_1R = _resnet_layer(conv5_2R, Feature_layer4, dialated_conv1)
    conv4_2R = _resnet_layer(conv4_1R, Feature_layer4, dialated_conv1)
    conv4_2RU = _upscore_layer(conv4_2R)
    #conv4_2RU = UpSampling2D()(conv4_2R)
    
    conv3_1R = _resnet_layer(conv4_2RU, Feature_layer3, 1)
    conv3_2R = _resnet_layer(conv3_1R, Feature_layer3, 1)
    conv3_2RU = _upscore_layer(conv3_2R)
    #conv3_2RU = UpSampling2D()(conv3_2R)
    
    conv2_1R = _resnet_layer(conv3_2RU, Feature_layer2, 1)
    conv2_2R = _resnet_layer(conv2_1R, Feature_layer2, 1)
    conv2_2RU = _upscore_layer(conv2_2R)
    #conv2_2RU = UpSampling2D()(conv2_2R)

    conv1_1R = _resnet_layer(conv2_2RU, Feature_layer1, 1)
    conv1_2R = _resnet_layer(conv1_1R, Feature_layer1, 1)
    
    #output before prediction
    #size of features are size of input image
    return conv1_2R

    
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

    
def resunet(input_size, num_classes):
  
    inputs = Input((input_size, input_size, 1))
    coarse_features = _feature_extraction(inputs)
    interpolated_output = _interpolation(coarse_features)
    
    #final prediction
    upscore0 = Conv2D(num_classes,1,padding='SAME', activation='softmax')(interpolated_output)
    
    model = Model(inputs=[inputs], outputs=[upscore0])
    return model
