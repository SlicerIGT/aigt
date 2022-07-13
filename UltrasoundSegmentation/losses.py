import tensorflow as tf


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
