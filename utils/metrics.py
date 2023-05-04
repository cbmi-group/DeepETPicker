import torch
import torch.nn.functional as F


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


# segmentation metrics
def seg_metrics(y_pred, y_true, smooth=1e-7, isTrain=True, threshold=0.5, use_sigmoid=False):
    #comment out if your model contains a sigmoid or equivalent activation layer
    if use_sigmoid:
        y_pred = F.sigmoid(y_pred)
    y_pred = torch.where(y_pred < threshold, torch.zeros(1).cuda(), torch.ones(1).cuda())

    #flatten label and prediction tensors
    y_pred = flatten(y_pred)
    y_true = flatten(y_true)
        
    tp = (y_true * y_pred).sum(-1)
    fp = ((1 - y_true) * y_pred).sum(-1)
    fn = (y_true * (1 - y_pred)).sum(-1)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    iou = (tp + smooth) / (tp + fn + fp + smooth)    
    f1 = 2 * (precision*recall) / (precision + recall + smooth)

    mean_precision = precision.mean()
    mean_recall = recall.mean()
    mean_iou = iou.mean()
    mean_f1 = f1.mean()

    # for training, ouput mean metrics
    if isTrain:
        return mean_precision.item(), mean_recall.item(), mean_iou.item(), mean_f1.item()
    # for testing, output metrics array by class with threshold
    else:
        return precision.detach().cpu().numpy(), recall.detach().cpu().numpy(), iou.detach().cpu().numpy(), f1.detach().cpu().numpy()


def seg_metrics_2d(y_pred, y_true, smooth=1e-7, isTrain=True, threshold=0.5, use_sigmoid=False):
    # comment out if your model contains a sigmoid or equivalent activation layer
    if use_sigmoid:
        y_pred = F.sigmoid(y_pred)
    y_pred = torch.where(y_pred < threshold, torch.zeros(1).cuda(), torch.ones(1).cuda())

    # flatten label and prediction tensors
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    tp = (y_true * y_pred).sum(-1)
    fp = ((1 - y_true) * y_pred).sum(-1)
    fn = (y_true * (1 - y_pred)).sum(-1)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    iou = (tp + smooth) / (tp + fn + fp + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)

    # for training, ouput mean metrics
    if isTrain:
        return precision.item(), recall.item(), iou.item(), f1.item()
    # for testing, output metrics array by class with threshold
    else:
        return precision.detach().cpu().numpy(), recall.detach().cpu().numpy(), iou.detach().cpu().numpy(), f1.detach().cpu().numpy()
