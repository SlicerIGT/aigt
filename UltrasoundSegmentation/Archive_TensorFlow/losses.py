import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Resizing, Rescaling


class WeightedCategoricalCrossEntropy(tf.keras.losses.Loss):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    Variables:
        class_weights: numpy array of shape (C,) where C is the number of classes
    """
    def __init__(self, class_weights, name="weighted_categorical_cross_entropy"):
        super().__init__(name=name)
        self.class_weights = tf.keras.backend.variable(class_weights)

    def call(self, y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        loss = y_true * tf.keras.backend.log(y_pred) * self.class_weights
        loss = -tf.keras.backend.sum(loss, -1)
        return loss


class BCELoss(tf.keras.losses.Loss):
    def __init__(self, name="bce_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        e = tf.keras.backend.epsilon()
        # Clip to prevent NaNs and Infs
        y_pred = tf.clip_by_value(y_pred, e, 1 - e)
        # Calc
        term_0 = y_true * tf.math.log(y_pred + e)
        term_1 = (1 - y_true) * tf.math.log(1 - y_pred + e)
        loss = -tf.math.reduce_mean(term_0 + term_1, axis=0)
        return loss


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-5, name="dice_loss"):
        super().__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        intersect = tf.math.reduce_sum(y_pred * y_true)
        predicted_sum = tf.math.reduce_sum(y_pred * y_pred)
        gt_sum = tf.math.reduce_sum(y_true * y_true)

        loss = (2 * intersect + self.smooth) / (gt_sum + predicted_sum + self.smooth)
        loss = 1 - loss
        return loss


class BCEDiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-5, name="bce_dice_loss"):
        super().__init__(name=name)
        self.smooth = smooth
        self.bce = BCELoss()
        self.dice = DiceLoss(smooth=smooth)

    def call(self, y_true, y_pred):
        loss = self.bce(y_true, y_pred) + self.dice(y_true, y_pred)
        return loss


class GGNetLoss(tf.keras.losses.Loss):
    def __init__(self, img_size=(128, 128), n_layers=4, alpha=1, beta=10, smooth=1e-5, name="gg_net_loss"):
        super().__init__(name=name)
        self.img_size = img_size
        self.n_layers = n_layers
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.dice = DiceLoss(smooth=smooth)
        self.bce = BCELoss()
        self.mse = tf.keras.losses.MeanSquaredError()

    def set_bd_label(self, y_true_bd):
        self.y_true_bd = y_true_bd

    def call(self, y_true, y_pred):
        # Calculate bd module loss
        bd_loss = 0
        for i in range(self.n_layers):
            # Get segmentation and boundary ground truths
            resize_factor = int(tf.math.pow(2, i + 1))
            y_true_seg = Resizing(self.img_size[0] // resize_factor, self.img_size[1] // resize_factor)(y_true)
            y_true_bd = Resizing(self.img_size[0] // resize_factor, self.img_size[1] // resize_factor)(self.y_true_bd)
            # Calculate segmentation loss
            layer_seg_loss = self.dice(y_true_seg, y_pred[i + 1][1]) - self.bce(y_true_seg, y_pred[i + 1][1])
            # Calculate boundary loss
            layer_bd_loss = self.mse(y_true_bd, y_pred[i + 1][0])
            # Add to bd_loss
            bd_loss += self.alpha * layer_seg_loss + self.beta * layer_bd_loss

        # Output segmentation loss
        out_seg_loss = self.dice(y_true, y_pred[0]) - self.bce(y_true, y_pred[0])

        # Total model loss
        loss = bd_loss + out_seg_loss
        return loss
