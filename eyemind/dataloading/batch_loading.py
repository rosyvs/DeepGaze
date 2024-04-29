
import random
import torch


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


def random_collate_fn(sequence_length, batch,  min_seq=1.0, max_seq=1.0, flag=-180, replace_na_flag=False, replacement=0):
    X, fix_y = zip(*batch)

    bs = len(X) # batch size
    fs = X[0].shape[-1] # features
    if min_seq != max_seq:
        sequence_length = random.randrange(int(min_seq*sequence_length), int(max_seq*sequence_length))
    X_batched = torch.zeros((bs,sequence_length,fs))
    fix_y_batched = torch.zeros((bs, sequence_length))
    X_mask = torch.ones_like(X)
    fix_y_mask = torch.ones_like(fix_y)
    for i in range(bs):
        full_sl = X[i].shape[0]
        start_ind = random.randrange(0,full_sl-sequence_length)
        end_ind = start_ind + sequence_length 
        X_batched[i] = X[i][start_ind:end_ind,:]
        fix_y_batched[i] = fix_y[i][start_ind:end_ind]
        # detect flagged and NA values in X and fix_y and compute a elementwise mask to return (0 = flagged, 1 = not flagged)
    # mask is elementwise, 0 if flagged, 1 if not flagged
    X_mask[X_batched == flag or torch.isnan(X_batched)] = 0
    fix_y_mask[fix_y_batched == flag or torch.isnan(fix_y_batched)] = 0
    # also na NaN and inf valeus add to mask

    # if replace_na_flag is True, replace masked values with replacement value using the mask
    if replace_na_flag:
        X_batched[X_mask==0] = replacement
        fix_y_batched[fix_y_mask==0] = replacement
    
    return (X_batched,X_mask), (fix_y_batched, fix_y_mask)
    
def random_multitask_collate_fn(sequence_length, batch, min_seq=1.0, max_seq=1.0, flag=-180, replace_na_flag=False, replacement=0):
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
    # masks
    X_mask[X_batched == flag or torch.isnan(X_batched)] = 0
    fix_y_mask[fix_y_batched == flag or torch.isnan(fix_y_batched)] = 0
    X2_mask[X2_batched == flag or torch.isnan(X2_batched)] = 0
    # replacements
    if replace_na_flag:
        X_batched[X_mask==0] = replacement
        fix_y_batched[fix_y_mask==0] = replacement
        X2_batched[X2_mask==0] = replacement
    return (X_batched, X_mask), (fix_y_batched, fix_y_mask),(X2_batched, X2_mask), cl_y_batched.float()

#TODO: mask/flag stuff from here on
def random_multilabel_multitask_collate_fn(sequence_length, batch, min_seq=1.0, max_seq=1.0, flag=-180):
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
        # compute masks
        X_mask = torch.tensor([torch.any(t==flag) for t in X])
        fix_y_mask = torch.tensor([torch.any(t==flag) for t in fix_y])
        # also na NaN and inf valeus add to mask   
        X_mask = torch.logical_or(X_mask, torch.isnan(X).any(dim=(1,2)))
        fix_y_mask = torch.logical_or(fix_y_mask, torch.isnan(fix_y).any(dim=(1,2)))
        X2_mask = torch.tensor([torch.any(t==flag) for t in X2_batched])
        X2_mask = torch.logical_or(X2_mask, torch.isnan(X2_batched).any(dim=(1,2)))
        seq_y_mask = torch.tensor([torch.any(t==flag) for t in seq_y])
        seq_y_mask = torch.logical_or(seq_y_mask, torch.isnan(seq_y).any(dim=(1,2)))
    return (X_batched, X_mask), (fix_y_batched, fix_y_mask), (seq_y_batched, seq_y_mask), (X2_batched, X2_mask), cl_y_batched.float()




# full-sequence collate functions
def variable_length_random_collate_fn(sequence_length, batch, max_sequence_length):
    X, fix_y = zip(*batch)
    bs = len(X)
    fs = X[0].shape[-1]
    lens = [x.shape[0] for x in X] # sequence lengths
    max_sequence_length = min(max_sequence_length, max(lens)) # if max_sequence_length is greater than the longest sequence, set it to the longest sequence
    X_batched = torch.zeros((bs,max_sequence_length,fs))
    fix_y_batched = torch.zeros((bs, max_sequence_length))
    pad_mask = torch.zeros((batch_size, max_sequence_length))

    for i in range(bs):
        full_sl = X[i].shape[0]
        if full_sl > max_sequence_length:
            start_ind = random.randrange(0,full_sl-max_sequence_length)
            end_ind = start_ind + max_sequence_length 
            X_batched[i] = X[i][start_ind:end_ind,:]
            fix_y_batched[i] = fix_y[i][start_ind:end_ind]
            pad_mask[i, :] = 1
        else:
            X_batched[i, :full_sl] = X[i]
            fix_y_batched[i, :full_sl] = fix_y[i]
            pad_mask[i, :full_sl] = 1
    return (X_batched, pad_mask),fix_y_batched
    
def variable_length_random_multitask_collate_fn(max_sequence_length, batch, CL_ratio = [0.2, 0.4]):
    """
    Takes variable length scanpaths and forms batch using full sequence 
    (if exceeding max_sequence_length) selects a random part of max_sequence_length
    for CL a subsequence of length CL_ratio[0] to CL_ratio[1] * full sequence length is selected
    # TODO: this incomplete, need to do mask and return it
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
    # TODO: this incomplete, need to do mask and return it

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



    # BATCH LOADERS #

    def predictive_coding_batch(X_batch, pc_seq_length, label_length, pred_length, offset=None): 
    # TODO: use offset ot select random segment if not None
    # TODO: deal w na and nan flag here? 
    # for the vanilla encoder-decoder models label_length should be 0 for consistency w old implementation
    assert pc_seq_length + pred_length <= X_batch.shape[1], f'pc_seq_length: {pc_seq_length} + pred_length: {pred_length} must be <= X_batch.shape[1]: {X_batch.shape[1]}'
    X_seq = X_batch[:,:pc_seq_length,:]
    Y_seq = X_batch[:,pc_seq_length-label_length:pc_seq_length+pred_length,:] # taken from later in the sequence, this is the target
    return X_seq, Y_seq

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




def predictive_coding_batch_variable_length(X_batch, X_pad_mask, label_length, pred_length): 
    # here X_pc is allowed to be variable length
    # X_pad_mask is a mask of the same shape as X_batch, with 1s where there is data and 0s where there is padding
    # get maximum sequence length
    max_seq_len = X_pad_mask.sum(dim=1).max().item()-pred_length
    batch_size = X_batch.shape[0]
    X_pc = torch.zeros((batch_size, max_seq_len, X_batch.shape[2]))
    Y_pc = torch.zeros((batch_size, label_length+pred_length, X_batch.shape[2]))
    X_pc_mask = torch.zeros((batch_size, max_seq_len))
    for x, pad in zip(X_batch, X_pad_mask):
        assert x[0].shape[0] == x[1].shape[0], f'X_batch and X_pad_mask must have the same length'
        # select x where pad is 1
        xpc = x[pad==1][:-pred_length]
        ypc = x[pad==1][-(pred_length+label_length):]
        # re-pad to max_seq_len with padding at start of sequence #TODO: use flag value for these? 
        pad_mask = torch.cat([torch.zeros((max_seq_len-xpc.shape[0])), torch.ones(xpc.shape[0])], dim=0)
        xpc = torch.cat([torch.zeros((max_seq_len-xpc.shape[0], xpc.shape[1])), xpc], dim=0)
        X_pc[i] = xpc
        Y_pc[i] = ypc
        X_pc_mask[i] = pad_mask
    return (X_pc,X_pc_mask) , Y_pc 