import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleLoss(nn.Module):
    """Loss function that can handle multi-scale outputs from the model"""
    def __init__(self, base_criterion=nn.L1Loss(), weights=None):
        super(MultiScaleLoss, self).__init__()
        self.base_criterion = base_criterion
        self.weights = weights or [0.7, 0.2, 0.1]  # Default weights for 3 scales
    
    def forward(self, pred, target):
        # If prediction is a list (multi-scale), compute weighted loss
        if isinstance(pred, list):
            total_loss = 0
            
            # Ensure we have weights for each output
            weights = self.weights
            if len(weights) != len(pred):
                weights = [1.0/len(pred)] * len(pred)
            
            # Calculate loss for each scale
            for i, p in enumerate(pred):
                loss = self.base_criterion(p, target)
                total_loss += weights[i] * loss
                
            return total_loss
        else:
            # Single output, just use base criterion
            return self.base_criterion(pred, target)
