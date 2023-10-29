from typing import Sequence

import torch
from torch import nn
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

def get_class_weights(class_ratio_or_counts: Sequence[float]):
    # Inverse of class frequencies. 
    #The weight for each class is computed by dividing the total number of samples by the product of 
    # the number of classes and the number of samples in each class.  
    numerator=sum(class_ratio_or_counts)
    n_classes=len(class_ratio_or_counts)
    weights=tuple([numerator/(n_classes*x) for x in class_ratio_or_counts])
    return weights