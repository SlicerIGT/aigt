import numpy as np
import scipy.ndimage
import warnings
import tensorflow as tf
from tensorflow.keras import backend as K

# String constants to avoid spelling errors

TRUE_POSITIVE_RATE  = "true_positive_rate"
RECALL              = "true_positive_rate"
SENSITIVITY         = "true_positive_rate"
FALSE_POSITIVE_RATE = "false_positive_rate"
FALLOUT             = "false_positive_rate"
SPECIFICITY         = "specificity"
PRECISION           = "precision"
DICE                = "dice"
JACCARD             = "jaccard"
ACCURACY            = "accuracy"
FSCORE              = "f_score"


def dilate_stack(segmentation_data, iterations):
    return np.array([scipy.ndimage.binary_dilation(y, iterations=iterations) for y in segmentation_data])


def compute_evaluation_metrics(prediction, groundtruth, acceptable_margin_mm=0.0, mm_per_pixel=1.0):
    """
    Computes evaluation metrics related to overlap
    :param prediction: np.array(x, y, z, c), c=0 for background, c=1 for foreground
    :param groundtruth: np.array(x, y, z, 0), 0 if background, 1 if foreground
    :param acceptable_margin_mm: positives this far from TP do not count as false positive
    :param mm_per_pixel: Obtain this value from image geometry
    :return: A dict() of results. Names are available as constants.
    """

    # num_slices = float(groundtruth.shape[0])

    acceptable_margin_pixel = int(acceptable_margin_mm / mm_per_pixel)

    actual_pos_map  = groundtruth[:, :, :, 0]
    actual_neg_map = 1.0 - actual_pos_map
    accept_pos_map = dilate_stack(groundtruth[:, :, :, 0], acceptable_margin_pixel)

    true_pos_map    = np.minimum(prediction[:, :, :, 1], actual_pos_map)
    true_neg_map    = np.minimum(prediction[:, :, :, 0], actual_neg_map)
    false_pos_map   = np.maximum(prediction[:, :, :, 1] - accept_pos_map, 0.0)
    false_neg_map   = np.maximum(prediction[:, :, :, 0] - actual_neg_map, 0.0)

    true_pos_total  = np.sum(true_pos_map, dtype="float64")
    true_neg_total  = np.sum(true_neg_map, dtype="float64")
    false_pos_total = np.sum(false_pos_map, dtype="float64")
    false_neg_total = np.sum(false_neg_map, dtype="float64")

    actual_pos_total = true_pos_total + false_neg_total
    actual_neg_total = true_neg_total + false_pos_total
    predict_pos_total = true_pos_total + false_pos_total

    # If there is no actual positive, then the sensitivity is perfect

    if actual_pos_total <= 0.0:
        true_pos_rate = 1.0
    else:
        true_pos_rate = true_pos_total / actual_pos_total

    # If there is no actual negative, then specificity is perfect (we detect all negatives), so this should be 0.0

    if actual_neg_total <= 0.0:
        false_pos_rate =  0.0
    else:
        false_pos_rate = false_pos_total / actual_neg_total

    # If there is no positive prediction, then precision is perfect.

    if predict_pos_total <= 0.0:
        precision = 1.0
    else:
        precision = true_pos_total / predict_pos_total

    specificity = true_neg_total / (true_neg_total + false_pos_total)

    # If there is no actual positive, and none is predicted, then dice should be perfect

    if (predict_pos_total + actual_pos_total) <= 0.0:
        dice = 1.0
    else:
        dice = 2 * true_pos_total / (predict_pos_total + actual_pos_total)

    jaccard  = true_pos_total / (true_pos_total + false_pos_total + false_neg_total)
    accuracy = (true_pos_total + true_neg_total) / (true_pos_total + false_pos_total + true_neg_total + false_neg_total)

    # If actual positive is zero, but some positive was predicted, then fcore is bad

    if (true_pos_rate + precision) <= 0.0:
        fscore = 0.0
    else:
        fscore   = 2 * true_pos_rate * precision / (true_pos_rate + precision)

    results = dict()
    results[TRUE_POSITIVE_RATE]  = true_pos_rate
    results[FALSE_POSITIVE_RATE] = false_pos_rate
    results[SPECIFICITY]         = specificity
    results[PRECISION]           = precision
    results[DICE]                = dice
    results[JACCARD]             = jaccard
    results[ACCURACY]            = accuracy
    results[FSCORE]              = fscore

    return results


def compute_roc(roc_thresholds, prediction_data, groundtruth_data, acceptable_margin_mm, mm_per_pixel):
    predictive_values    = np.zeros(len(roc_thresholds))  # Distance from ROC diagonal
    false_positive_rates = np.zeros(len(roc_thresholds))
    true_positive_rates  = np.zeros(len(roc_thresholds))
    metrics_dicts = dict()

    for i in range(len(roc_thresholds)):
        threshold = roc_thresholds[i]
        prediction_thresholded = np.zeros(prediction_data.shape)
        prediction_thresholded[:, :, :, 1][prediction_data[:, :, :, 1] >= threshold] = 1.0
        prediction_thresholded[:, :, :, 0][prediction_data[:, :, :, 1] < threshold] = 1.0
        # prediction_thresholded[prediction_thresholded < threshold] = 0.0
        metrics = compute_evaluation_metrics(
            prediction_thresholded, groundtruth_data, acceptable_margin_mm=acceptable_margin_mm,
            mm_per_pixel=mm_per_pixel)

        false_positive_rates[i] = metrics[FALSE_POSITIVE_RATE]
        true_positive_rates[i]  = metrics[TRUE_POSITIVE_RATE]
        crossprod = np.cross((1.0, 1.0), (false_positive_rates[i], true_positive_rates[i]))
        predictive_values[i] = np.linalg.norm(crossprod) / np.linalg.norm([1.0, 1.0])
        metrics_dicts[i] = metrics

    area = 0.0
    for i in range(len(roc_thresholds)):
        if i == len(roc_thresholds) - 1:
            area = area + (1.0 - false_positive_rates[i]) * true_positive_rates[i]
        else:
            area = area + (false_positive_rates[i + 1] - false_positive_rates[i]) * true_positive_rates[i]

    best_threshold_index = np.argmax(predictive_values)

    return metrics_dicts, best_threshold_index, area

def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def jaccard_coef(y_true, y_pred):
    y_true = y_true[:, :, :, :2]
    
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    jac = (intersection + 1.) / (union - intersection + 1.)
    return K.mean(jac)

def threshold_binarize(x, threshold=0.5):
    ge = tf.greater_equal(x, tf.constant(threshold))
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y

def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def dice_coef(y_true, y_pred, smooth=1.):
    y_true = y_true[:, :, :, :2]
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
                K.sum(y_true_f) + K.sum(y_pred_f) + smooth)