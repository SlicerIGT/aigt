"""Implementation of fuzzy segmentation metrics."""

import numpy as np
import pandas as pd
import torch


class FuzzyMetrics:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.count = 0
        self.cm = torch.zeros((self.num_classes, 4), dtype=torch.float32)
        self.accuracy = torch.zeros(self.num_classes, dtype=torch.float32)
        self.precision = torch.zeros(self.num_classes, dtype=torch.float32)
        self.sensitivity = torch.zeros(self.num_classes, dtype=torch.float32)
        self.specificity = torch.zeros(self.num_classes, dtype=torch.float32)
        self.f1_score = torch.zeros(self.num_classes, dtype=torch.float32)
        self.dice = torch.zeros(self.num_classes, dtype=torch.float32)
        self.iou = torch.zeros(self.num_classes, dtype=torch.float32)

    def compute_confusion_matrix(self, pred, target, softmax=False):
        """Computes the fuzzy confusion matrix for image segmentations.

            TP = sum(min(pred, target))
            FP = sum(max(pred - target, 0))
            TN = sum(min(1 - pred, 1 - target))
            FN = sum(max(target - pred, 0))

        Args:
            pred (Tensor): One-hot encoded Tensor of activated model outputs with shape (1, N, H, W)
            target (Tensor): One-hot encoded Tensor of ground truth labels with shape (1, N, H, W)
        """
        if softmax:
            pred = torch.softmax(pred, dim=1)

        self.count += 1
        for class_idx in range(self.num_classes):
            pred_class = pred[:, class_idx, :, :]
            target_class = target[:, class_idx, :, :]

            tp = torch.sum(torch.minimum(pred_class, target_class))
            fp = torch.sum(torch.maximum(pred_class - target_class, torch.tensor(0., dtype=torch.float32)))
            tn = torch.sum(torch.minimum(1 - pred_class, 1 - target_class))
            fn = torch.sum(torch.maximum(target_class - pred_class, torch.tensor(0., dtype=torch.float32)))

            self.cm[class_idx] += torch.tensor([tp, fp, tn, fn])

    def compute_accuracy(self):
        """Computes a fuzzy implementation of the accuracy metric.

            Acc = (TP + TN) / (TP + FP + TN + FN)
        """
        for class_idx in range(self.num_classes):
            tp, fp, tn, fn = self.cm[class_idx]
            self.accuracy[class_idx] += (tp + tn) / (tp + fp + tn + fn)

    def compute_precision(self):
        """Computes a fuzzy implementation of the precision metric.

            Pre = TP / (TP + FP)
        """
        for class_idx in range(self.num_classes):
            tp, fp, _, _ = self.cm[class_idx]
            self.precision[class_idx] += tp / (tp + fp)

    def compute_sensitivity(self):
        """Computes a fuzzy implementation of the sensitivity (recall/TPR) metric.

            Sen = TP / (TP + FN)
        """
        for class_idx in range(self.num_classes):
            tp, _, _, fn = self.cm[class_idx]
            self.sensitivity[class_idx] += tp / (tp + fn)

    def compute_specificity(self):
        """Computes a fuzzy implementation of the specificity (TNR) metric.

            Spe = TN / (TN + FP)
        """
        for class_idx in range(self.num_classes):
            _, fp, tn, _ = self.cm[class_idx]
            self.specificity[class_idx] += tn / (tn + fp)

    def compute_f1_score(self):
        """Computes a fuzzy implementation of the F1 measure.

            F1 = 2 * Pre * Sen / (Pre + Sen)
        """
        for class_idx in range(self.num_classes):
            tp, fp, _, fn = self.cm[class_idx]
            pre = tp / (tp + fp)
            sen = tp / (tp + fn)
            self.f1_score[class_idx] += 2 * pre * sen / (pre + sen)

    def compute_dice(self):
        """Computes a fuzzy implementation of the Dice coefficient.

            Dice = 2 * TP / (2 * TP + FP + FN)
        """
        for class_idx in range(self.num_classes):
            tp, fp, _, fn = self.cm[class_idx]
            self.dice[class_idx] += 2 * tp / (2 * tp + fp + fn)

    def compute_iou(self):
        """Computes a fuzzy implementation of the IoU (intersection over union/Jaccard) metric.

            IoU = TP / (TP + FP + FN)
        """
        for class_idx in range(self.num_classes):
            tp, fp, _, fn = self.cm[class_idx]
            self.iou[class_idx] += tp / (tp + fp + fn)
    
    def update_metrics(self, pred, target, softmax=False):
        """Computes all metrics for the given prediction and target tensors."""
        self.compute_confusion_matrix(pred, target, softmax=softmax)
        self.compute_accuracy()
        self.compute_precision()
        self.compute_sensitivity()
        self.compute_specificity()
        self.compute_f1_score()
        self.compute_dice()
        self.compute_iou()
    
    def get_mean_metrics_by_class(self):
        """Returns the mean for all metrics, split by class."""
        avg_acc_by_class = (self.accuracy / self.count).numpy()
        avg_pre_by_class = (self.precision / self.count).numpy()
        avg_sen_by_class = (self.sensitivity / self.count).numpy()
        avg_spe_by_class = (self.specificity / self.count).numpy()
        avg_f1_by_class = (self.f1_score / self.count).numpy()
        avg_dice_by_class = (self.dice / self.count).numpy()
        avg_iou_by_class = (self.iou / self.count).numpy()
        return avg_acc_by_class, avg_pre_by_class, avg_sen_by_class, avg_spe_by_class, \
               avg_f1_by_class, avg_dice_by_class, avg_iou_by_class
    
    def get_total_mean_metrics(self):
        """Returns the mean for all metrics, averaged over all classes."""
        avg_acc_by_class, avg_pre_by_class, avg_sen_by_class, avg_spe_by_class, \
        avg_f1_by_class, avg_dice_by_class, avg_iou_by_class = self.get_mean_metrics_by_class()
        avg_acc = np.nanmean(avg_acc_by_class[1:])
        avg_pre = np.nanmean(avg_pre_by_class[1:])
        avg_sen = np.nanmean(avg_sen_by_class[1:])
        avg_spe = np.nanmean(avg_spe_by_class[1:])
        avg_f1 = np.nanmean(avg_f1_by_class[1:])
        avg_dice = np.nanmean(avg_dice_by_class[1:])
        avg_iou = np.nanmean(avg_iou_by_class[1:])
        return avg_acc, avg_pre, avg_sen, avg_spe, avg_f1, avg_dice, avg_iou

    def get_metrics_as_dataframe(self):
        """Adds all metrics to a Pandas dataframe."""
        avg_acc_by_class, avg_pre_by_class, avg_sen_by_class, avg_spe_by_class, \
        avg_f1_by_class, avg_dice_by_class, avg_iou_by_class = self.get_mean_metrics_by_class()
        avg_acc, avg_pre, avg_sen, avg_spe, avg_f1, avg_dice, avg_iou = self.get_total_mean_metrics()

        metrics_df = pd.DataFrame(
            columns=[class_idx for class_idx in range(self.num_classes)],
            index=["accuracy", "precision", "sensitivity", "specificity", "f1_score", "dice", "iou"]
        )
        metrics_df.loc["accuracy"] = avg_acc_by_class
        metrics_df.loc["precision"] = avg_pre_by_class
        metrics_df.loc["sensitivity"] = avg_sen_by_class
        metrics_df.loc["specificity"] = avg_spe_by_class
        metrics_df.loc["f1_score"] = avg_f1_by_class
        metrics_df.loc["dice"] = avg_dice_by_class
        metrics_df.loc["iou"] = avg_iou_by_class
        metrics_df["total"] = [avg_acc, avg_pre, avg_sen, avg_spe, avg_f1, avg_dice, avg_iou]

        return metrics_df


if __name__ == "__main__":
    metrics = FuzzyMetrics(num_classes=2)

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
    # metrics.update_metrics(pred, target)
    # print(metrics.get_total_mean_metrics())

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
    # metrics.update_metrics(pred, target)
    # print(metrics.get_total_mean_metrics())

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
    # metrics.update_metrics(pred, target)
    # print(metrics.get_total_mean_metrics())

    # Test 4: fuzzy predictions
    pred = torch.tensor(np.array(
        [[[[0.3, 0],
           [0, 0]],
          [[0.7, 1],
           [0, 0]]]]
    ))
    target = torch.tensor(np.array(
        [[[[0, 0],
           [0, 0]],
          [[1, 1],
           [1, 1]]]]
    ))
    metrics.update_metrics(pred, target)
    print(metrics.get_total_mean_metrics())
    print(metrics.get_metrics_as_dataframe())
