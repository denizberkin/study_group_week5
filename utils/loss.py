import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, preds, masks, smooth=1e-6):
        preds = torch.sigmoid(preds)  # Apply sigmoid to predictions
        preds = preds.contiguous()
        masks = masks.contiguous()

        intersection = (preds * masks).sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + smooth)
        return 1 - dice.mean()
    
    def forward(self, preds, masks):
        bce_loss = self.bce(preds, masks)
        dice_loss = self.dice_loss(preds, masks)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
