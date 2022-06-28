import tensorflow as tf


class WeightedCategoricalCrossEntropy(tf.keras.losses.Loss):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        class_weights: numpy array of shape (C,) where C is the number of classes
    """
    def __init__(self, class_weights, name="weighted_categorical_cross_entropy"):
        super().__init__(name=name)
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.math.reduce_sum(y_pred, axis=-1, keepdims=True)
        # Clip to prevent NaN's and Inf's
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # Calc
        loss = y_true * tf.math.log(y_pred) * self.class_weights
        loss = -tf.math.reduce_sum(loss, axis=-1)
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
    def __init__(self, class_weights, smooth=1e-5, name="bce_dice_loss"):
        super().__init__(name=name)
        self.smooth = smooth
        self.bce = WeightedCategoricalCrossEntropy(class_weights)
        self.dice = DiceLoss(smooth=smooth)

    def call(self, y_true, y_pred):
        loss = self.bce(y_true, y_pred) + self.dice(y_true, y_pred)
        return loss
