import numpy as np
import tensorflow as tf


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


class WeightedCategoricalCrossEntropy(tf.keras.losses.Loss):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        class_weights: numpy array of shape (C,) where C is the number of classes
    """
    def __init__(self, class_weights, name="weighted_categorical_cross_entropy"):
        super().__init__(name=name)
        self.class_weights = class_weights
        self.bce = BCELoss()

    def call(self, y_true, y_pred):
        # Calculate BCE
        bce_loss = self.bce(y_true, y_pred)
        # Apply weights
        weight_vector = y_true * self.class_weights[0] + (1 - y_true) * self.class_weights[1]
        weighted_bce = weight_vector * bce_loss
        # Calculate loss
        loss = tf.math.reduce_mean(weighted_bce)
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
