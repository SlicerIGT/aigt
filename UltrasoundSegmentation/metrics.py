"""Implementation of fuzzy segmentation metrics."""

import torch


def confusion_matrix(pred, target, sigmoid=False, softmax=False):
    """Computes the fuzzy confusion matrix for image segmentations.

        TP = sum(min(pred, target))
        FP = sum(max(pred - target, 0))
        TN = sum(min(1 - pred, 1 - target))
        FN = sum(max(target - pred, 0))

    Args:
        pred (Tensor): One-hot encoded Tensor of model outputs with shape (1, N, H, W)
        target (Tensor): One-hot encoded Tensor of ground truth labels with shape (1, N, H, W)
        sigmoid (bool, optional): Use sigmoid activation. Defaults to False.
        softmax (bool, optional): Use softmax activation. Defaults to False.

    Returns:
        tuple of Tensors: true positives, false positives, true negatives, false negatives
    """
    assert not (sigmoid and softmax), "sigmoid and softmax cannot both be True"

    if sigmoid:
        pred = torch.sigmoid(pred)
    if softmax:
        pred = torch.softmax(pred)
    
    tp = torch.sum(torch.minimum(pred, target))
    fp = torch.sum(torch.maximum(pred - target, torch.tensor(0, dtype=torch.int8)))
    tn = torch.sum(torch.minimum(1 - pred, 1 - target))
    fn = torch.sum(torch.maximum(target - pred, torch.tensor(0, dtype=torch.int8)))

    assert round((tp + fp + tn + fn).item()) == torch.numel(pred), \
        f"TP + FP + TN + FN = {tp + fp + tn + fn}, but should be {torch.numel(pred)}"

    return tp, fp, tn, fn


def fuzzy_accuracy(pred, target, sigmoid=False, softmax=False):
    """Computes a fuzzy implementation of the accuracy metric.

        Acc = (TP + TN) / (TP + FP + TN + FN)

    Args:
        pred (Tensor): One-hot encoded Tensor of model outputs with shape (1, N, H, W)
        target (Tensor): Tensor of ground truth labels with shape (1, 1, H, W), 
                         note that the labels are not one-hot encoded
        sigmoid (bool, optional): Use sigmoid activation. Defaults to False.
        softmax (bool, optional): Use softmax activation. Defaults to False.

    Returns:
        float: accuracy
    """
    tp, fp, tn, fn = confusion_matrix(pred, target, sigmoid, softmax)
    acc = (tp + tn) / (tp + fp + tn + fn)
    return acc.item()


def fuzzy_precision(pred, target, sigmoid=False, softmax=False):
    """Computes a fuzzy implementation of the precision metric.

        Acc = TP / (TP + FP)

    Args:
        pred (Tensor): One-hot encoded Tensor of model outputs with shape (1, N, H, W)
        target (Tensor): Tensor of ground truth labels with shape (1, 1, H, W), 
                         note that the labels are not one-hot encoded
        sigmoid (bool, optional): Use sigmoid activation. Defaults to False.
        softmax (bool, optional): Use softmax activation. Defaults to False.

    Returns:
        float: precision
    """
    tp, fp, _, _ = confusion_matrix(pred, target, sigmoid, softmax)
    precision = tp / (tp + fp)
    return precision.item()


def fuzzy_sensitivity(pred, target, sigmoid=False, softmax=False):
    """Computes a fuzzy implementation of the sensitivity (recall/TPR) metric.

        Sen = TP / (TP + FN)

    Args:
        pred (Tensor): One-hot encoded Tensor of model outputs with shape (1, N, H, W)
        target (Tensor): Tensor of ground truth labels with shape (1, 1, H, W), 
                         note that the labels are not one-hot encoded
        sigmoid (bool, optional): Use sigmoid activation. Defaults to False.
        softmax (bool, optional): Use softmax activation. Defaults to False.

    Returns:
        float: sensitivity
    """
    tp, _, _, fn = confusion_matrix(pred, target, sigmoid, softmax)
    sen = tp / (tp + fn)
    return sen.item()


def fuzzy_specificity(pred, target, sigmoid=False, softmax=False):
    """Computes a fuzzy implementation of the specificity (TNR) metric.

        Spe = TN / (TN + FP)

    Args:
        pred (Tensor): One-hot encoded Tensor of model outputs with shape (1, N, H, W)
        target (Tensor): Tensor of ground truth labels with shape (1, 1, H, W), 
                         note that the labels are not one-hot encoded
        sigmoid (bool, optional): Use sigmoid activation. Defaults to False.
        softmax (bool, optional): Use softmax activation. Defaults to False.

    Returns:
        float: specificity
    """
    _, fp, tn, _ = confusion_matrix(pred, target, sigmoid, softmax)
    spe = tn / (tn + fp)
    return spe.item()


def fuzzy_f1_score(pred, target, sigmoid=False, softmax=False):
    """Computes a fuzzy implementation of the F1 measure.

        F1 = 2 * Pre * Sen / (Pre + Sen)

    Args:
        pred (Tensor): One-hot encoded Tensor of model outputs with shape (1, N, H, W)
        target (Tensor): Tensor of ground truth labels with shape (1, 1, H, W), 
                         note that the labels are not one-hot encoded
        sigmoid (bool, optional): Use sigmoid activation. Defaults to False.
        softmax (bool, optional): Use softmax activation. Defaults to False.

    Returns:
        float: F1 score
    """
    tp, fp, _, fn = confusion_matrix(pred, target, sigmoid, softmax)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    f1 = 2 * pre * sen / (pre + sen)
    return f1.item()


def fuzzy_dice(pred, target, sigmoid=False, softmax=False):
    """Computes a fuzzy implementation of the Dice coefficient.

        Dice = 2 * TP / (2 * TP + FP + FN)

    Args:
        pred (Tensor): One-hot encoded Tensor of model outputs with shape (1, N, H, W)
        target (Tensor): Tensor of ground truth labels with shape (1, 1, H, W), 
                         note that the labels are not one-hot encoded
        sigmoid (bool, optional): Use sigmoid activation. Defaults to False.
        softmax (bool, optional): Use softmax activation. Defaults to False.

    Returns:
        float: Dice coefficient
    """
    tp, fp, _, fn = confusion_matrix(pred, target, sigmoid, softmax)
    dice = 2 * tp / (2 * tp + fp + fn)
    return dice.item()


def fuzzy_iou(pred, target, sigmoid=False, softmax=False):
    """Computes a fuzzy implementation of the IoU (intersection over union) metric.

        IoU = TP / (TP + FP + FN)

    Args:
        pred (Tensor): One-hot encoded Tensor of model outputs with shape (1, N, H, W)
        target (Tensor): Tensor of ground truth labels with shape (1, 1, H, W), 
                         note that the labels are not one-hot encoded
        sigmoid (bool, optional): Use sigmoid activation. Defaults to False.
        softmax (bool, optional): Use softmax activation. Defaults to False.

    Returns:
        float: IoU (Jaccard index)
    """
    tp, fp, _, fn = confusion_matrix(pred, target, sigmoid, softmax)
    iou = tp / (tp + fp + fn)
    return iou.item()


if __name__ == "__main__":
    import numpy as np

    # Test 1: prediction and target are the same
    pred = torch.tensor(np.array(
        [[[[0, 0],
           [0, 0]],
          [[1, 1],
           [1, 1]]]]
    ))
    target = torch.tensor(np.array(
        [[[[0, 0],
           [0, 0]],
          [[1, 1],
           [1, 1]]]]
    ))
    tp, fp, tn, fn = confusion_matrix(pred, target)
    print(tp, fp, tn, fn)
    print(fuzzy_dice(pred, target))

    # Test 2: prediction and target are completely different
    pred = torch.tensor(np.array(
        [[[[0, 0],
           [0, 0]],
          [[0, 0],
           [0, 0]]]]
    ))
    target = torch.tensor(np.array(
        [[[[0, 0],
           [0, 0]],
          [[1, 1],
           [1, 1]]]]
    ))
    tp, fp, tn, fn = confusion_matrix(pred, target)
    print(tp, fp, tn, fn)
    print(fuzzy_dice(pred, target))

    # Test 3: prediction is half correct
    pred = torch.tensor(np.array(
        [[[[0, 0],
           [0, 0]],
          [[1, 1],
           [0, 0]]]]
    ))
    target = torch.tensor(np.array(
        [[[[0, 0],
           [0, 0]],
          [[1, 1],
           [1, 1]]]]
    ))
    tp, fp, tn, fn = confusion_matrix(pred, target)
    print(tp, fp, tn, fn)
    print(fuzzy_dice(pred, target))

    # Test 4: fuzzy predictions
    pred = torch.tensor(np.array(
        [[[[0, 0],
           [0, 0]],
          [[0.5, 1],
           [0, 0]]]]
    ))
    target = torch.tensor(np.array(
        [[[[0, 0],
           [0, 0]],
          [[1, 1],
           [1, 1]]]]
    ))
    tp, fp, tn, fn = confusion_matrix(pred, target)
    print(tp, fp, tn, fn)
    print(fuzzy_dice(pred, target))
