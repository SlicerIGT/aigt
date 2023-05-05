def soft_iou(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / union
    return iou

'''
# Make prediction
output = model(image)
pred = torch.sigmoid(output)  # Apply sigmoid to obtain fuzzy segmentation mask

# Calculate soft IoU
iou_score = soft_iou(pred, label)

'''