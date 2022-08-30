from matplotlib.pyplot import flag
import numpy as np
import numpy.ma as ma
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

class StandardScaler():

    def __init__(self, mean=[-0.698, -1.940], std=[4.15, 3.286], flag=-180):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.flag = flag
    
    def __call__(self, data):
        cols = data.shape[-1]
        assert(cols==len(self.mean))
        assert(cols==len(self.std))
        mx = ma.masked_values(data,self.flag)
        mx_stand = (mx - self.mean) / self.std
        return mx_stand.filled(self.flag)

    def inverse_transform(self, data):
        # tensors
        if isinstance(data, torch.Tensor):
            mask = data == self.flag
            data = data * torch.tensor(self.std) + torch.tensor(self.mean)
            data[mask] = self.flag
            return data
        elif isinstance(data, np.ndarray):
            mx = ma.masked_values(data, self.flag)
            mx_scaled = mx*self.std + self.mean
            return mx_scaled.filled(self.flag)
        else:
            raise TypeError("Data should be a torch tensor or a numpy array")