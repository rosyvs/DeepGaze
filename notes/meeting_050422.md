## Notes Prior

### Table 4 Experiment Sequence length vs capacity:

It might be a good idea to train these for longer since the loss curves are still decreasing and auroc is increasing. Especially for the larger capacity models, it's possible that they would perform better on longer sequences if they were trained longer.

### Code Restructuring

1. Need to not store all the data modules and models since these will take up a lot of memory

I think taking the hyperparam experiment and making it so that it creates and runs experiments one at a time passing in the needed parameters will be the best way.

2. Data Module 

Look at examples in pytorch lightning. I think the main problem is reading through all of the files to get the labels. This should probably be done when getting items and not before hand 

Want to make it so the distributed data loader works

3. LightningModule Model 

This could be refactored so that the only things passed to the init function are hyperparameters and then it will create the encoder and decoder from these within the module. This way the hyperparameters can be dealt with nicely and will get outputted nicely.

4. Metrics

AUROC metric might cause problems when the dataset is large because it has to store all the outputs. Figure out whether this is causing memory issues

5. Profiling
The code is running slowly on the gpus and additionally might have memory leak issues. How can I see the memory usage and run the code across multiple gpus? Should each experiment run in its own task on the cluster?


## Meeting 5/19/22
### Diagram to explain all the parameter training and architectures 
Detail every experiment in a table that makes the terminology all clear
### Kfolds are the same for fixation id and inference tests

### Nested cross val for fixation and inference


## Meeting 6/2/22

### Pre-Meeting

Detail the experiments:
Fixation Identification:
- Cross Validation Split
- Tested Sequence Length (250, 500) and Hidden Dimension (128, 256)
- Trained max 40 Epochs
- Model splits sequences into sequence_length size and encodes then decoder auto-regressively predicts the fixation label
- Save checkpoints of best performing models

Comprehension (Rote_X):
- Nested Cross Validation Group Stratified Split (4 outer and inner folds)
- Initialized encoder with trained encoder weights from the fixation identification task
- Tested freezing and tuning encoder.
- Choose best hyperparameter setting for outer fold based on average performance across the inner folds 
- Model: splits scanpaths into sequence_length size and each gets the same label, encoder -> classification head (multi-layer perceptron), predicts label for subsequence, Average predicted labels for subsequences to make prediction for whole scanpath

Detail the splitting strategy and why

Things to work on:
1. Check why fold2 isn't showing up. Need more logging for this probably
2. Deal with padding and masking so that the model doesn't take into account sequences that are mostly padded
3. Check the correct way to use ddp with raytune so that you can train trials with multiple gpus

Results:
1. Fixation Identification: 
Cross Validation: Shouldn't cause overfitting since it is a different task. Can train it on different splits if needed but this way we have an upper bound and know if it isn't working. 
Didn't test for shorter sequence length than 250 because a sequence length of 100 is 1.7s of reading time which seems quite short to use. Sequence lengths to times 100: 1.7s , 250: 4.25s, 500: 8.5s
Seems like a larger hidden dimension helps. Might be good to test even larger size.

2. Comprehension: Rote_X
Nested Cross Validation: 4 outer and 4 inner folds. Tested two configurations: freezing and not freezing encoder
Uses trained encoder from Fixation Identification. Breaks up full scanpaths into sequences of 250 and predicts on each then takes the average to get the final result.
Validation loss and AUROC don't improve.
Training loss and training AUROC improve: possibly overfitting. Probably need to analyze the predictions to see what is happening. Could it be learning some pattern in the data like length or that multiple sequences belonging to the same scanpath have the same label?

### Meeting Notes

- First sequence only
- Random Sequence
- Use the outer folds splits for 