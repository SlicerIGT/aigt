import numpy as np
import scipy.ndimage


# String constants to avoid spelling errors

FALSE_POSITIVE_PREDICTION_MM = "false_positive_prediction_mm2"
TRUE_NEGATIVE_AREA_MM = "true_negative_area_mm2"
TRUE_NEGATIVE_AREA_PERCENT = "true_negative_area_percent"
TRUE_POSITIVE_PREDICTION_MM = "true_positive_prediction_mm2"
TRUE_POSITIVE_AREA_MM = "true_positive_area_mm2"
TRUE_POSITIVE_AREA_PERCENT = "true_positive_area_percent"


def dilate_stack(segmentation_data, iterations):
    return np.array([scipy.ndimage.binary_dilation(y, iterations=iterations) for y in segmentation_data])

def compute_evaluation_metrics(prediction, groundtruth, acceptable_margin_mm=1.0, mm_per_pixel=1.0):
    num_slices = groundtruth.shape[0]

    acceptable_margin_pixel = int(acceptable_margin_mm / mm_per_pixel)
    acceptable_region = dilate_stack(groundtruth[:, :, :, 0], acceptable_margin_pixel)
    true_pos_prediction = np.minimum(groundtruth[:, :, :, 0], prediction[:, :, :, 1])
    not_acceptable_region = 1 - acceptable_region
    false_pos_prediction = np.minimum(not_acceptable_region, prediction[:, :, :, 1])

    fpp = np.sum(false_pos_prediction[:, :, :])
    tna = np.sum(not_acceptable_region[:, :, :])
    tpp = np.sum(true_pos_prediction)
    tpa = np.sum(groundtruth[:, :, :, 0])

    results = dict()
    results[FALSE_POSITIVE_PREDICTION_MM] = fpp * mm_per_pixel * mm_per_pixel / num_slices
    results[TRUE_NEGATIVE_AREA_MM] = tna * mm_per_pixel * mm_per_pixel / num_slices
    results[TRUE_NEGATIVE_AREA_PERCENT] = (tna - fpp) / tna * 100
    results[TRUE_POSITIVE_PREDICTION_MM] = tpp * mm_per_pixel * mm_per_pixel / num_slices
    results[TRUE_POSITIVE_AREA_MM] = tpa * mm_per_pixel * mm_per_pixel / num_slices
    results[TRUE_POSITIVE_AREA_PERCENT] = tpp / tpa * 100

    return results
