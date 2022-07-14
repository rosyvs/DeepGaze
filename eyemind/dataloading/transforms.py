import numpy as np
import torch
from torchvision.transforms import Normalize

class LimitSequenceLength(object):
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
    
    def __call__(self, sequence, random=False):
        if len(sequence) > self.sequence_length:
            # Remove part of data
            if random:
                start_idx = random.randint(0,len(sequence) - self.sequence_length)
                sequence = sequence[start_idx:start_idx+self.sequence_length]
            else:
                sequence = sequence[:self.sequence_length]    
        else:
            # Pad data
            if len(sequence.shape) == 2:
                padded_data = np.zeros((self.sequence_length,sequence.shape[1]))
            elif len(sequence.shape) == 1:
                padded_data = np.zeros((self.sequence_length,))
            else:
                raise ValueError(f"Tensor shape of {sequence.shape} is not allowed")
            padded_data[:len(sequence)] = sequence
            sequence = padded_data
        return sequence 

class ToTensor(object):
    def __init__(self):
        pass
    def __call__(self, v):
        return torch.tensor(v).float()

class Standardizer(Normalize):

    def inverse_transform(self, data):
        
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean 