from matplotlib.pyplot import flag
import numpy as np
import numpy.ma as ma
import torch
import pandas as pd
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

class GazeScaler():

    def __init__(self, mean=[-0.698, -1.955], std=[4.113, 3.234], flag=-180):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.flag = flag
    
    def __call__(self, data):
        cols = data.shape[-1]
        assert(cols==len(self.mean))
        assert(cols==len(self.std))
        mx = ma.masked_values(data,self.flag)
        mx_stand = (mx - self.mean) / self.std
        return mx_stand.filled(self.flag) # even sfter scaling flag is still 180

    def inverse_transform(self, data):
        # tensors
        if isinstance(data, torch.Tensor):
            mask = data == self.flag
            data = data * torch.tensor(self.std, device=data.device) + torch.tensor(self.mean, device=data.device)
            data[mask] = self.flag
            return data
        elif isinstance(data, np.ndarray):
            mx = ma.masked_values(data, self.flag)
            mx_scaled = mx*self.std + self.mean
            return mx_scaled.filled(self.flag)
        else:
            raise TypeError("Data should be a torch tensor or a numpy array")

class StandardScaler():
    def __init__(self, mean=0.0, std=1.0):
        self.mean = np.array(mean)
        self.std = np.array(std)
    def __call__(self, data):
        mx_stand = (data - self.mean) / self.std
        return mx_stand
    def inverse_transform(self, data):
        # tensors
        if isinstance(data, torch.Tensor):
            data = data * torch.tensor(self.std, device=data.device) + torch.tensor(self.mean, device=data.device)
            data[mask] = self.flag
            return data
        elif isinstance(data, (np.ndarray, list, pd.Series)):
            mx_scaled = data*self.std + self.mean
            return mx_scaled
        else:
            raise TypeError("Data should be a torch tensor or a numpy array")

class FlagReplacer():
    # replace values "flagged" during data prep by replacing with a new set value
    # replacement value is 0 by default, so apply after Scaler() to avoid scaling the new flag value
    def __init__(self, flag=-180, replacement=0):
        self.flag = flag
        self.replacement = replacement
    def __call__(self, data):
        data[data==self.flag] = self.replacement
        return data

class Pooler():
    # sequence pooling functions for pooling transformer encoder output to flatten time dimension
    def __init__(self, pool_method):
        self.pool_fn = self.get_pooler(pool_method)
    def __call__(self, *args):
        return self.pool_fn(self, *args)
    def mean_pool_logits(self, logits, mask=None):
        return logits.mean(dim=1)
    def masked_mean_pool_logits(self, logits, mask=None):
        if mask is None:
            mask = torch.ones(logits.shape[0], logits.shape[1]) # reduce to mean if mask is not provided
        return (logits*mask.unsqueeze(2)).sum(dim=1) / mask.sum(dim=1).unsqueeze(1)
    def final_pos_pool_logits(self,logits, mask=None):
        if mask is None:
            mask = torch.ones(logits.shape[0], logits.shape[1])
        final_pos = torch.stack([torch.nonzero(mask[i,:], as_tuple=True)[0][-1] for i in range(mask.shape[0])])
        return torch.stack([logits[i,final_pos[i],:] for i in range(logits.shape[0])])
    def get_pooler(self,pool_method):
        if pool_method == 'mean':
            return Pooler.mean_pool_logits
        elif pool_method == 'masked_mean':
            return Pooler.masked_mean_pool_logits
        elif pool_method == 'final_pos':
            return Pooler.final_pos_pool_logits
        elif pool_method is None:
            return lambda x: x  # identity function
        else:
            raise ValueError(f"Pooling method {pool_method} not recognized")