
import random
import torch


def collate_fn_pad(batch, sequence_length):
    '''
    Pads batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    X, y = zip(*batch)
    X_lengths = torch.tensor([ t.shape[0] for t in X ])
    y_lengths = torch.tensor([ t.shape[0] for t in y ])
    ## padd
    X_padded = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=-181.)
    y_padded = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=-181.)
    # Split
    torch.split(X_padded, sequence_length, )
    ## compute mask
    return X_padded, y_padded, X_lengths, y_lengths

def split_collate_fn(sequence_length, batch, contrastive=False):
    '''
    Takes variable length sequences, splits them each into 
    subsequences of sequence_length, and returns tensors:
    
    Args:
        batch: List[Tuples(Tensor(X), Tensor(y))] Contains a list of the returned items from dataset
        sequence_length: int lengths of the subsequences

    Returns:
        X: Tensor shape (bs, sequence_length, *)
        y: Tensor shape (bs, sequence_length)
    '''
    X, fix_y = zip(*batch)
    # Splits each example into tensors with sequence length and drops last in case it is a different length
    X_splits = [torch.stack(torch.split(t, sequence_length, dim=0)[:-1], dim=0) for t in X]
    fix_y_splits = [torch.stack(torch.split(t, sequence_length, dim=0)[:-1], dim=0) for t in fix_y]
    
    X = torch.cat(X_splits, dim=0)
    fix_y = torch.cat(fix_y_splits, dim=0)
    if contrastive:
        cl_y = torch.cat([torch.ones(X_split.shape[0])*i for i, X_split in enumerate(X_splits)], dim=0)
        return X, fix_y, cl_y
    return X, fix_y

def multitask_collate_fn(sequence_length, batch, contrastive=False):
    '''
    Takes variable length sequences, splits them each into 
    subsequences of sequence_length, and returns tensors:
    
    Args:
        batch: List[Tuples(Tensor(X), Tensor(y))] Contains a list of the returned items from dataset
        sequence_length: int lengths of the subsequences

    Returns:
        X: Tensor shape (bs, sequence_length, *)
        y: Tensor shape (bs, sequence_length)
    '''
    X, fix_y = zip(*batch)
    # Splits each example into tensors with sequence length and drops last in case it is a different length
    X_splits = [torch.stack(torch.split(t, sequence_length, dim=0)[:-1], dim=0) for t in X]
    fix_y_splits = [torch.stack(torch.split(t, sequence_length, dim=0)[:-1], dim=0) for t in fix_y]
    X_fix = torch.cat(X_splits, dim=0)
    fix_y = torch.cat(fix_y_splits, dim=0)
    if contrastive:
        X1, X2, y = contrastive_batch(X,sequence_length)
        return X_fix, fix_y, X1, X2, y
    return X_fix, fix_y

    
def predictive_coding_batch(X_batch, input_length, pred_length, label_length):
    X_seq = X_batch[:,:input_length,:]
    y_seq = X_batch[:,input_length-label_length:input_length+pred_length,:]
    return X_seq, y_seq

def reconstruction_batch(X_batch, label_length):
    decoder_inp = torch.zeros_like(X_batch)
    decoder_inp[:,:label_length,:] = X_batch[:,:label_length,:]
    return decoder_inp

def contrastive_batch(X, sequence_length, sub_length=(0.4,0.7)):
    n = len(X)
    fs = X[0][0].shape[-1]
    s1 = random.randrange(int(sequence_length * sub_length[0]), int(sequence_length * sub_length[1]))
    s2 = random.randrange(int(sequence_length * sub_length[0]), int(sequence_length * sub_length[1]))
    x1 = torch.zeros((n, s1, fs))
    x2 = torch.zeros((n, s2, fs))
    y = torch.zeros(n)
    for i in range(n):
    # get x1
        sl = X[i].shape[0]
        try:
            x1_start = random.randrange(0, sl - s1)
        except:
            x1_start = 0
        x1[i, :, :] = X[i][x1_start:x1_start + s1, :]

        if random.random() > 0.5:
            # Get x2 from the same sequence
            j = i
            y[i] = 1
        else:
            # Get x2 from different sequence
            j = i
            y[i] = 0
            while j == i:
                j = random.randrange(0, n)

        try:
            x2_start = random.randrange(0, sl - s2)
        except:
            x2_start = 0

        x2[i, :, :] = X[j][x2_start:x2_start + s2, :]

    return x1.float(), x2.float(), y.float()

def fixation_batch(input_length, label_length, pred_length, X, y, padding=-1.):
    '''
    Takes variable length sequences, splits them each into 
    subsequences of sequence_length, and returns tensors

    Args:
        input_length: (int) Length of encoder input
        pred_length: (int) Length of predictions for decoder output
        label_length: (int) Length of input labels to decoder for start
        X: (Tensor) Sequence data (bs, sequence_length_scanpath, 2)
        y: (Tensor) Fixation labels (bs, sequence_length_scanpath)
    
    Returns:
        X: Gaze data (bs*sequence_length_scanpath/input_length, input_length, 2)
        decoder_inp: Input to incoder with first label_length of each batch as the actual fixation data and the rest masked 
        targets: Fixation labels for each subsequence
    '''

    targets = y[:, -pred_length:]
    if padding == 0:
        decoder_inp = torch.zeros_like(y)
    else: 
        decoder_inp = torch.ones_like(y) * padding
    decoder_inp[:, :label_length] = y[:,:label_length]
    return decoder_inp, targets    

