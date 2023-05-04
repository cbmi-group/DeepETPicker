from torch import nn as nn


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


# Dice loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, args=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.use_softmax = args.use_softmax
        self.use_sigmoid = args.use_sigmoid


    def forward(self, outputs, targets):
        # flatten label and prediction tensors
        outputs = flatten(outputs)
        targets = flatten(targets)

        intersection = (outputs * targets).sum(-1)
        dice = (2. * intersection + self.smooth) / (outputs.sum(-1) + targets.sum(-1) + self.smooth)
        return 1 - dice.mean()
