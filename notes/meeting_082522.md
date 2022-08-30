## Updates

1. Ran multitask for 100 epochs for 500 sequence length input, 100 sequence length prediction. 
Results:
Fixation ID didn't improve AUROC ~0.6.  

Contrastive Learning did train well: ~81% accuracy

Updating the predictive coding task and reconstruction task to mask out the flag values as they are outliers and throw off the prediction greatly

2. Ran predictive coding task for transformer model for 10 epochs at 500 sequence length, 100 sequence length prediction.

Results:
Looks like it is training well. Updating metric so that we can see the actual distances between the predictions and the ground truth
Also going to plot a few to get a good sense of what they are looking like

3. Labeling regressions. Should saccades be labeled or just NaN. and then will test a model on the regression task

## Todos

1. Inverse scaling so outputs are scaled back to input range and we can plot predictions vs ground truths

2. Fixation identification plotting

3. Smoothness of fixation identification. Can we change the loss function to encourage smoothness?

4. Get alpine to work with gpus

## Notes
First pass fixation, regressive fixation, 

