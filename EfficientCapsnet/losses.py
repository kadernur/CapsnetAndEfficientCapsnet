import torch
from torch import nn
class MarginLoss(nn.Module):
   
    def __init__(self,
                 present_max: float = 0.9,
                 absent_min: float = 0.1,
                 absent_scale: float = 0.5) -> None:
        super(MarginLoss, self).__init__(name="MarginLoss")
        self.present_max = present_max
        self.absent_min = absent_min
        self.absent_scale = absent_scale

    def forward(self, labels, digit_probs):
        
        assert labels.shape is not digit_probs.shape
        zeros = torch.zeros_like(labels, dtype= torch.float32)
        present_losses = labels * torch.square(
            torch.maximum(zeros, self.present_max - digit_probs))
        absent_losses = (1 - labels) * torch.square(
            torch.maximum(zeros, digit_probs - self.absent_min))
        losses = present_losses + self.absent_scale * absent_losses
        return torch.reduce_sum(losses, axis=-1, name="total_loss")