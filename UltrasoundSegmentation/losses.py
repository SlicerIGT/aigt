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
