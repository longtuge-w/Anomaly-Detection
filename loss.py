import torch
import torch.nn as nn


class TotalSavingLoss(nn.Module):
    def __init__(self, cost_matrix, allowable_time_window=60):
        super(TotalSavingLoss, self).__init__()
        self.cost_matrix = cost_matrix
        self.allowable_time_window = allowable_time_window

    def forward(self, predictions, targets):
        # predictions: [B, 5]
        # targets: [B, 5]
        
        # Convert predictions to binary values (0 or 1)
        predictions = (predictions > 0.5).float()
        
        # Calculate true positives (TP), false negatives (FN), and false positives (FP)
        tp = (predictions * targets).sum(dim=0)
        fn = ((1 - predictions) * targets).sum(dim=0)
        fp = (predictions * (1 - targets)).sum(dim=0)
        
        # Calculate the lead time for true positives
        lead_time = torch.where(predictions * targets == 1, self.allowable_time_window, torch.tensor(0.0))
        
        # Calculate the total saving for each component
        total_saving = ((self.cost_matrix[:, 0] - self.cost_matrix[:, 1]) * lead_time / self.allowable_time_window).sum() - fn * self.cost_matrix[:, 0] - fp * self.cost_matrix[:, 2]
        
        # Normalize the total saving by the number of components
        normalized_total_saving = total_saving / len(self.cost_matrix)
        
        # Negate the normalized total saving to convert it into a loss value
        loss = -normalized_total_saving
        
        return loss