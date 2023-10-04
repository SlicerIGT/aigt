"""Implementation of fuzzy segmentation metrics."""

import numpy as np
import pandas as pd
import torch


class FuzzyMetrics:
    ACC_INDEX = 0
    PRE_INDEX = 1
    SEN_INDEX = 2
    SPE_INDEX = 3
    F1_INDEX = 4
    DICE_INDEX = 5
    IOU_INDEX = 6

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.counts = torch.zeros((7, self.num_classes), dtype=torch.int32)
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

        self.counts += 1
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
            acc = (tp + tn) / (tp + fp + tn + fn)
            if torch.isnan(acc):
                self.counts[self.ACC_INDEX, class_idx] -= 1
            else:
                self.accuracy[class_idx] += acc

    def compute_precision(self):
        """Computes a fuzzy implementation of the precision metric.

            Pre = TP / (TP + FP)
        """
        for class_idx in range(self.num_classes):
            tp, fp, _, _ = self.cm[class_idx]
            pre = tp / (tp + fp)
            if torch.isnan(pre):
                self.counts[self.PRE_INDEX, class_idx] -= 1
            else:
                self.precision[class_idx] += pre

    def compute_sensitivity(self):
        """Computes a fuzzy implementation of the sensitivity (recall/TPR) metric.

            Sen = TP / (TP + FN)
        """
        for class_idx in range(self.num_classes):
            tp, _, _, fn = self.cm[class_idx]
            sen = tp / (tp + fn)
            if torch.isnan(sen):
                self.counts[self.SEN_INDEX, class_idx] -= 1
            else:
                self.sensitivity[class_idx] += sen

    def compute_specificity(self):
        """Computes a fuzzy implementation of the specificity (TNR) metric.

            Spe = TN / (TN + FP)
        """
        for class_idx in range(self.num_classes):
            _, fp, tn, _ = self.cm[class_idx]
            spe = tn / (tn + fp)
            if torch.isnan(spe):
                self.counts[self.SPE_INDEX, class_idx] -= 1
            else:
                self.specificity[class_idx] += spe

    def compute_f1_score(self):
        """Computes a fuzzy implementation of the F1 measure.

            F1 = 2 * Pre * Sen / (Pre + Sen)
        """
        for class_idx in range(self.num_classes):
            tp, fp, _, fn = self.cm[class_idx]
            pre = tp / (tp + fp)
            sen = tp / (tp + fn)
            f1 = 2 * pre * sen / (pre + sen)
            if torch.isnan(f1):
                self.counts[self.F1_INDEX, class_idx] -= 1
            else:
                self.f1_score[class_idx] += f1

    def compute_dice(self):
        """Computes a fuzzy implementation of the Dice coefficient.

            Dice = 2 * TP / (2 * TP + FP + FN)
        """
        for class_idx in range(self.num_classes):
            tp, fp, _, fn = self.cm[class_idx]
            dice = 2 * tp / (2 * tp + fp + fn)
            if torch.isnan(dice):
                self.counts[self.DICE_INDEX, class_idx] -= 1
            else:
                self.dice[class_idx] += dice

    def compute_iou(self):
        """Computes a fuzzy implementation of the IoU (intersection over union/Jaccard) metric.

            IoU = TP / (TP + FP + FN)
        """
        for class_idx in range(self.num_classes):
            tp, fp, _, fn = self.cm[class_idx]
            iou = tp / (tp + fp + fn)
            if torch.isnan(iou):
                self.counts[self.IOU_INDEX, class_idx] -= 1
            else:
                self.iou[class_idx] += iou
    
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
        avg_acc_by_class = (self.accuracy / self.counts[self.ACC_INDEX]).numpy()
        avg_pre_by_class = (self.precision / self.counts[self.PRE_INDEX]).numpy()
        avg_sen_by_class = (self.sensitivity / self.counts[self.SEN_INDEX]).numpy()
        avg_spe_by_class = (self.specificity / self.counts[self.SPE_INDEX]).numpy()
        avg_f1_by_class = (self.f1_score / self.counts[self.F1_INDEX]).numpy()
        avg_dice_by_class = (self.dice / self.counts[self.DICE_INDEX]).numpy()
        avg_iou_by_class = (self.iou / self.counts[self.IOU_INDEX]).numpy()
        return avg_acc_by_class, avg_pre_by_class, avg_sen_by_class, avg_spe_by_class, \
               avg_f1_by_class, avg_dice_by_class, avg_iou_by_class
    
    def get_total_mean_metrics(self):
        """Returns the mean for all metrics, averaged over all classes (including background)."""
        avg_acc_by_class, avg_pre_by_class, avg_sen_by_class, avg_spe_by_class, \
        avg_f1_by_class, avg_dice_by_class, avg_iou_by_class = self.get_mean_metrics_by_class()
        avg_acc = np.mean(avg_acc_by_class)
        avg_pre = np.mean(avg_pre_by_class)
        avg_sen = np.mean(avg_sen_by_class)
        avg_spe = np.mean(avg_spe_by_class)
        avg_f1 = np.mean(avg_f1_by_class)
        avg_dice = np.mean(avg_dice_by_class)
        avg_iou = np.mean(avg_iou_by_class)
        return avg_acc, avg_pre, avg_sen, avg_spe, avg_f1, avg_dice, avg_iou

    def get_metrics_as_dataframe(self):
        """Adds all metrics to a Pandas dataframe."""
        avg_acc_by_class, avg_pre_by_class, avg_sen_by_class, avg_spe_by_class, \
        avg_f1_by_class, avg_dice_by_class, avg_iou_by_class = self.get_mean_metrics_by_class()
        avg_acc, avg_pre, avg_sen, avg_spe, avg_f1, avg_dice, avg_iou = self.get_total_mean_metrics()

        metrics_df = pd.DataFrame(
            columns=[class_idx for class_idx in range(self.num_classes)],
            index=["accuracy_fuzzy", "precision_fuzzy", "sensitivity_fuzzy", 
                   "specificity_fuzzy", "f1_score_fuzzy", "dice_fuzzy", "iou_fuzzy"]
        )
        metrics_df.loc["accuracy_fuzzy"] = avg_acc_by_class
        metrics_df.loc["precision_fuzzy"] = avg_pre_by_class
        metrics_df.loc["sensitivity_fuzzy"] = avg_sen_by_class
        metrics_df.loc["specificity_fuzzy"] = avg_spe_by_class
        metrics_df.loc["f1_score_fuzzy"] = avg_f1_by_class
        metrics_df.loc["dice_fuzzy"] = avg_dice_by_class
        metrics_df.loc["iou_fuzzy"] = avg_iou_by_class
        metrics_df["average"] = [avg_acc, avg_pre, avg_sen, avg_spe, avg_f1, avg_dice, avg_iou]

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
