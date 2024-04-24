
import random
import torch

# NOTE: UNUSED
# def collate_fn_pad(batch, sequence_length):
#     '''
#     Pads batch of variable length

#     note: it converts things ToTensor manually here since the ToTensor transform
#     assume it takes in images rather than arbitrary tensors.
#     '''
#     X, y = zip(*batch)
#     X_lengths = torch.tensor([ t.shape[0] for t in X ])
#     y_lengths = torch.tensor([ t.shape[0] for t in y ])
#     ## padd
#     X_padded = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=-181.)
#     y_padded = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=-181.)
#     # Split
#     torch.split(X_padded, sequence_length, ) #TODO: this doing antyhing?? not returned...
#     ## compute mask # TODO: why no mask here?
#     return X_padded, y_padded, X_lengths, y_lengths

# used in SequenceToSequenceDataModule
def split_collate_fn(sequence_length, batch, contrastive=False):
    '''
    Takes variable length sequences, splits them each into 
    subsequences of sequence_length, and returns tensors:
    NOTE: NON RANDOM
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

# # NOTE: unused
# def multitask_collate_fn(sequence_length, batch, contrastive=False):
#     '''
#     Takes variable length sequences, splits them each into 
#     subsequences of sequence_length, and returns tensors:
    
#     Args:
#         batch: List[Tuples(Tensor(X), Tensor(y))] Contains a list of the returned items from dataset
#         sequence_length: int lengths of the subsequences

#     Returns:
#         X: Tensor shape (bs, sequence_length, *)
#         y: Tensor shape (bs, sequence_length)
#     '''
#     X, fix_y = zip(*batch)
#     # Splits each example into tensors with sequence length and drops last in case it is a different length
#     X_splits = [torch.stack(torch.split(t, sequence_length, dim=0)[:-1], dim=0) for t in X]
#     fix_y_splits = [torch.stack(torch.split(t, sequence_length, dim=0)[:-1], dim=0) for t in fix_y]
#     X_fix = torch.cat(X_splits, dim=0)
#     fix_y = torch.cat(fix_y_splits, dim=0)
#     if contrastive:
#         X1, X2, y = contrastive_batch(X,sequence_length)
#         return X_fix, fix_y, X1, X2, y
#     return X_fix, fix_y

def random_collate_fn(sequence_length, batch,  min_seq=1.0, max_seq=1.0):
    X, fix_y = zip(*batch)
    bs = len(X)
    fs = X[0].shape[-1]
    if min_seq != max_seq:
        sequence_length = random.randrange(int(min_seq*sequence_length), int(max_seq*sequence_length))
    X_batched = torch.zeros((bs,sequence_length,fs))
    fix_y_batched = torch.zeros((bs, sequence_length))
    for i in range(bs):
        full_sl = X[i].shape[0]
        start_ind = random.randrange(0,full_sl-sequence_length)
        end_ind = start_ind + sequence_length 
        X_batched[i] = X[i][start_ind:end_ind,:]
        fix_y_batched[i] = fix_y[i][start_ind:end_ind]
    return X_batched, fix_y_batched
    
def random_multitask_collate_fn(sequence_length, batch, min_seq=1.0, max_seq=1.0):
    """
    Takes variable length scanpaths and selects a random part of SET sequence length and batches them
    """
    X, fix_y = zip(*batch)
    bs = len(X)
    fs = X[0].shape[-1]
    if min_seq != max_seq:
        sequence_length = random.randrange(int(min_seq*sequence_length), int(max_seq*sequence_length))
    X_batched = torch.zeros((bs,sequence_length,fs))
    fix_y_batched = torch.zeros((bs, sequence_length))
    X2_batched = torch.zeros((bs,sequence_length,fs))
    cl_y_batched = torch.randint(0,2,(bs,))
    for i in range(bs):
        full_sl = X[i].shape[0]
        start_ind = random.randrange(0,full_sl-sequence_length)
        end_ind = start_ind + sequence_length 
        X_batched[i] = X[i][start_ind:end_ind,:] # choose random interval within sequence
        fix_y_batched[i] = fix_y[i][start_ind:end_ind]
        if cl_y_batched[i] == 0:
            j = i
            while j == i:
                j = random.randrange(0,bs)
            full_sl = X[j].shape[0]
            start_ind = random.randrange(0,full_sl-sequence_length)
            end_ind = start_ind + sequence_length
            X2_batched[i] = X[j][start_ind:end_ind,:]
        else:
            start_ind = random.randrange(0,full_sl-sequence_length)
            end_ind = start_ind + sequence_length
            X2_batched[i] = X[i][start_ind:end_ind,:]
    return X_batched, fix_y_batched, X2_batched, cl_y_batched.float()

def random_multilabel_multitask_collate_fn(sequence_length, batch, min_seq=1.0, max_seq=1.0):
    """
    Takes variable length scanpaths and selects a random part of sequence length and batches them
    For multilabel data (sequence and fixation labels) for use with class SequenceToMultilabelDataModule
    """
    X, Y  = zip(*batch) # todo perhaps unpack y first then split into fix and seq? 
    fix_y, seq_y = zip(*Y)
    bs = len(X)
    fs = X[0].shape[-1]
    if min_seq != max_seq:
        sequence_length = random.randrange(int(min_seq*sequence_length), int(max_seq*sequence_length))
    X_batched = torch.zeros((bs,sequence_length,fs))
    fix_y_batched = torch.zeros((bs, sequence_length))
    X2_batched = torch.zeros((bs,sequence_length,fs))
    cl_y_batched = torch.randint(0,2,(bs,)) # randomly set each item in batch to have same or diff source for CL
    seq_y_batched=torch.zeros((bs,1)) # needs to be converted from tuple to tensor for batch
    for i in range(bs): # loop over batch
        full_sl = X[i].shape[0]
        start_ind = random.randrange(0,full_sl-sequence_length) # randomly choose sequence start
        end_ind = start_ind + sequence_length 
        X_batched[i] = X[i][start_ind:end_ind,:] # choose random interval within sequence
        fix_y_batched[i] = fix_y[i][start_ind:end_ind]
        seq_y_batched[i,] = seq_y[i]
        if cl_y_batched[i] == 0:
            j = i
            while j == i:
                j = random.randrange(0,bs) # choose another item from batch to pair this x with
            full_sl = X[j].shape[0]
            start_ind = random.randrange(0,full_sl-sequence_length)
            end_ind = start_ind + sequence_length
            X2_batched[i] = X[j][start_ind:end_ind,:]
        else: # simply choose another random subsequence from same seq (can overlap)
            start_ind = random.randrange(0,full_sl-sequence_length)
            end_ind = start_ind + sequence_length
            X2_batched[i] = X[i][start_ind:end_ind,:] 
    return X_batched, fix_y_batched, seq_y_batched, X2_batched, cl_y_batched.float()

def predictive_coding_batch(X_batch, sequence_length, pred_length, label_length): 
    # NOTE: this is actually used already and assumes X is already trimmed to sequence_length
    # but I am confused about how this works when sequence_length+pred_length exceeds the length of the sequence
    full_sl = X_batch.shape[1]
    X_seq = X_batch[:,:sequence_length,:]
    y_seq = X_batch[:,sequence_length-label_length:sequence_length+pred_length,:] # taken from later in the sequence, this is the target
    return X_seq, y_seq

# NOTE: not used
def reconstruction_batch(X_batch, label_length): # not used
    decoder_inp = torch.zeros_like(X_batch)
    decoder_inp[:,:label_length,:] = X_batch[:,:label_length,:]
    return decoder_inp

# NOTE: doesnt get used
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


# NOTE: not used
def fixation_batch(input_length, label_length, pred_length, X, y, padding=-1.):
    '''
    Takes variable length sequences, splits them each into 
    subsequences of sequence_length, and returns tensors

    Args:
        input_length: (int) Length of encoder input
        pred_length: (int) Length of predictions for decoder output
        label_length: (int) Length of input labels to decoder for start # TODO: is this cheating? using earlier labels to predict future fix labels...
        X: (Tensor) Sequence data (bs, sequence_length_scanpath, 2)
        y: (Tensor) Fixation labels (bs, sequence_length_scanpath)
        padding: (float) Value to pad the decoder input with TODO: why not -180?
    
    Returns:
        decoder_inp: Gaze data (bs*sequence_length_scanpath/input_length, input_length, 2)
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






# full-sequence collate functions
def variable_length_random_collate_fn(sequence_length, batch,  min_seq=1.0, max_seq=1.0):
    X, fix_y = zip(*batch)
    bs = len(X)
    fs = X[0].shape[-1]
    if min_seq != max_seq:
        sequence_length = random.randrange(int(min_seq*sequence_length), int(max_seq*sequence_length))
    X_batched = torch.zeros((bs,sequence_length,fs))
    fix_y_batched = torch.zeros((bs, sequence_length))
    for i in range(bs):
        full_sl = X[i].shape[0]
        start_ind = random.randrange(0,full_sl-sequence_length)
        end_ind = start_ind + sequence_length 
        X_batched[i] = X[i][start_ind:end_ind,:]
        fix_y_batched[i] = fix_y[i][start_ind:end_ind]
    return X_batched, fix_y_batched
    
def variable_length_random_multitask_collate_fn(max_sequence_length, batch, CL_ratio = [0.2, 0.4]):
    """
    Takes variable length scanpaths and forms batch using full sequence 
    (if exceeding max_sequence_length) selects a random part of max_sequence_length
    for CL a subsequence of length CL_ratio[0] to CL_ratio[1] * full sequence length is selected
    """
    X, fix_y = zip(*batch)
    bs = len(X) # batch size
    fs = X[0].shape[-1] # feature size
    lens = [x.shape[0] for x in X] # sequence lengths
    max_sequence_length = min(max_sequence_length, max(lens)) # if max_sequence_length is greater than the longest sequence, set it to the longest sequence
    X_batched = torch.zeros((bs,max_sequence_length,fs))
    fix_y_batched = torch.zeros((bs, max_sequence_length))
    X2_batched = torch.zeros((bs,max_sequence_length,fs))
    cl_y_batched = torch.randint(0,2,(bs,))
    for i in range(bs):
        full_sl = X[i].shape[0]
        if full_sl > max_sequence_length:
            start_ind = random.randrange(0,full_sl-max_sequence_length)
            end_ind = start_ind + max_sequence_length 
            X_batched[i] = X[i][start_ind:end_ind,:] # choose random interval within sequence
            fix_y_batched[i] = fix_y[i][start_ind:end_ind]
        else:
            X_batched[i] = X[i]
            fix_y_batched[i] = fix_y[i]
        if cl_y_batched[i] == 0: # different source
            j = i
            while j == i:
                j = random.randrange(0,bs) # ransomly choose another in the batch
            full_sl = X[j].shape[0]
            subseq_len = random.randrange(int(CL_ratio[0]*full_sl), int(CL_ratio[1]*full_sl))
            if subseq_len > max_sequence_length:
                subseq_len = max_sequence_length
            start_ind = random.randrange(0,full_sl-subseq_len)
            end_ind = start_ind + sequence_length
            X2_batched[i] = X[j][start_ind:end_ind,:]
        else:
            start_ind = random.randrange(0,full_sl-sequence_length)
            end_ind = start_ind + sequence_length
            X2_batched[i] = X[i][start_ind:end_ind,:]
    return X_batched, fix_y_batched, X2_batched, cl_y_batched.float()

def variable_length_random_multilabel_multitask_collate_fn(sequence_length, batch, min_seq=1.0, max_seq=1.0):
    """
    Takes variable length scanpaths and selects a random part of sequence length and batches them
    For multilabel data (sequence and fixation labels) for use with class SequenceToMultilabelDataModule
    """
    X, Y  = zip(*batch) # todo perhaps unpack y first then split into fix and seq? 
    fix_y, seq_y = zip(*Y)
    bs = len(X)
    fs = X[0].shape[-1]
    if min_seq != max_seq:
        sequence_length = random.randrange(int(min_seq*sequence_length), int(max_seq*sequence_length))
    X_batched = torch.zeros((bs,sequence_length,fs))
    fix_y_batched = torch.zeros((bs, sequence_length))
    X2_batched = torch.zeros((bs,sequence_length,fs))
    cl_y_batched = torch.randint(0,2,(bs,)) # randomly set each item in batch to have same or diff source for CL
    seq_y_batched=torch.zeros((bs,1)) # needs to be converted from tuple to tensor for batch
    for i in range(bs): # loop over batch
        full_sl = X[i].shape[0]
        start_ind = random.randrange(0,full_sl-sequence_length) # randomly choose sequence start
        end_ind = start_ind + sequence_length 
        X_batched[i] = X[i][start_ind:end_ind,:] # choose random interval within sequence
        fix_y_batched[i] = fix_y[i][start_ind:end_ind]
        seq_y_batched[i,] = seq_y[i]
        if cl_y_batched[i] == 0:
            j = i
            while j == i:
                j = random.randrange(0,bs) # choose another item from batch to pair this x with
            full_sl = X[j].shape[0]
            start_ind = random.randrange(0,full_sl-sequence_length)
            end_ind = start_ind + sequence_length
            X2_batched[i] = X[j][start_ind:end_ind,:]
        else: # simply choose another random subsequence from same seq (can overlap)
            start_ind = random.randrange(0,full_sl-sequence_length)
            end_ind = start_ind + sequence_length
            X2_batched[i] = X[i][start_ind:end_ind,:] 
    return X_batched, fix_y_batched, seq_y_batched, X2_batched, cl_y_batched.float()