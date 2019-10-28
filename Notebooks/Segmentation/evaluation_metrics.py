import numpy as np
import scipy.ndimage
from random import sample

# String constants to avoid spelling errors

FALSE_POSITIVE_PREDICTION_MM = "false_positive_prediction_mm2"
TRUE_NEGATIVE_AREA_MM = "true_negative_area_mm2"
TRUE_NEGATIVE_AREA_PERCENT = "true_negative_area_percent"
TRUE_POSITIVE_PREDICTION_MM = "true_positive_prediction_mm2"
TRUE_POSITIVE_AREA_MM = "true_positive_area_mm2"
TRUE_POSITIVE_AREA_PERCENT = "true_positive_area_percent"
RECALL = "recall"
PRECISION = "precision"
FSCORE = "f_score"


def dilate_stack(segmentation_data, iterations):
    return np.array([scipy.ndimage.binary_dilation(y, iterations=iterations) for y in segmentation_data])

def compute_evaluation_metrics(prediction, groundtruth, acceptable_margin_mm=1.0, mm_per_pixel=1.0):
    """
    Computes evaluation metrics related to overlap
    :param prediction: np.array(x, y, z, c), c=0 for background, c=1 for foreground
    :param groundtruth: np.array(x, y, z, 0), 0 if background, 1 if foreground
    :param acceptable_margin_mm: positives this far from TP do not count as false positive
    :param mm_per_pixel: Obtain this value from image geometry
    :return: A dict() of results. Names are available as constants.
    """
    num_slices = groundtruth.shape[0]

    acceptable_margin_pixel = int(acceptable_margin_mm / mm_per_pixel)
    acceptable_region = dilate_stack(groundtruth[:, :, :, 0], acceptable_margin_pixel)
    true_pos_prediction = np.minimum(groundtruth[:, :, :, 0], prediction[:, :, :, 1])
    not_acceptable_region = 1 - acceptable_region
    false_pos_prediction = np.minimum(not_acceptable_region, prediction[:, :, :, 1])
    false_neg_prediction = np.maximum( (groundtruth[:, :, :, 0] - prediction[:, :, :, 1]), 0.0 )

    fpp = np.sum(false_pos_prediction[:, :, :])
    tna = np.sum(not_acceptable_region[:, :, :])
    tpp = np.sum(true_pos_prediction)
    tpa = np.sum(groundtruth[:, :, :, 0])
    fnp = np.sum(false_neg_prediction[:, :, :])
    recall = tpp / (tpp + fnp)
    precision = tpp / (tpp + fpp)

    '''
    print("***")
    print("fpp: {}".format(fpp))
    print("tpp: {}".format(tpp))
    print("fnp: {}".format(fnp))
    print("rec: {}".format(recall))
    print("pre: {}".format(precision))
    '''

    results = dict()
    results[FALSE_POSITIVE_PREDICTION_MM] = fpp * mm_per_pixel * mm_per_pixel / num_slices
    results[TRUE_NEGATIVE_AREA_MM] = tna * mm_per_pixel * mm_per_pixel / num_slices
    results[TRUE_NEGATIVE_AREA_PERCENT] = (tna - fpp) / tna * 100
    results[TRUE_POSITIVE_PREDICTION_MM] = tpp * mm_per_pixel * mm_per_pixel / num_slices
    results[TRUE_POSITIVE_AREA_MM] = tpa * mm_per_pixel * mm_per_pixel / num_slices
    results[TRUE_POSITIVE_AREA_PERCENT] = tpp / tpa * 100
    results[RECALL] = tpp / (tpp + fnp)
    results[PRECISION] = tpp / (tpp + fpp)
    results[FSCORE] = 2 * results[RECALL] * results[PRECISION] / (results[RECALL] + results[PRECISION])

    return results


def compute_roc(roc_thresholds, prediction_data, groundtruth_data, acceptable_margin_mm, mm_per_pixel):
    false_positives = np.zeros(len(roc_thresholds))
    true_positives = np.zeros(len(roc_thresholds))
    goodnesses = np.zeros(len(roc_thresholds))
    metrics_dicts = dict()

    for i in range(len(roc_thresholds)):
        threshold = roc_thresholds[i]
        prediction_thresholded = np.copy(prediction_data)
        prediction_thresholded[prediction_thresholded >= threshold] = 1.0
        prediction_thresholded[prediction_thresholded < threshold] = 0.0
        metrics = compute_evaluation_metrics(
            prediction_thresholded, groundtruth_data, acceptable_margin_mm=acceptable_margin_mm,
            mm_per_pixel=mm_per_pixel)
        true_negative_area_perc = metrics[TRUE_NEGATIVE_AREA_PERCENT]
        false_positives[i] = (100 - true_negative_area_perc) / 100.0
        true_positives[i] = metrics[TRUE_POSITIVE_AREA_PERCENT] / 100.0
        crossprod = np.cross((1.0, 1.0), (false_positives[i], true_positives[i]))
        goodnesses[i] = np.linalg.norm(crossprod) / np.linalg.norm([1.0, 1.0])
        metrics_dicts[i] = metrics

    area = 0.0
    for i in range(len(roc_thresholds)):
        if i == len(roc_thresholds) - 1:
            area = area + (1.0 - false_positives[i]) * true_positives[i]
        else:
            area = area + (false_positives[i + 1] - false_positives[i]) * true_positives[i]

    best_threshold_index = np.argmax(goodnesses)
    return true_positives, false_positives, best_threshold_index, area, metrics_dicts
