import numpy as np
import torch


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
            padded_data = np.zeros((self.sequence_length,sequence.shape[1]))
            padded_data[:len(sequence)] = sequence
            sequence = padded_data
        return sequence 

class ToTensor(object):
    def __init__(self):
        pass
    def __call__(self, v):
        return torch.tensor(v).float()