## Updates

1. The informer model performs much better on predictive coding task

2. Fixation prediction jumps all over the place during training and tends to either predict all fixations or no fixations. I'm adapting the sampling to equalize the number of examples and try to stabalize training.

3. Plotted fixation predictions to see what is happening. 

# Thoughts

1. Fixation Identification Debugging

Add number of unique fixations as prediction task to help it learn to separate fixations

Is it catching longer saccades but not shorter ones?

Does sample weighting have a large effect on training?

2. Other tasks

Can the informer be adapted to do the other tasks well?


# 

1. Make model capacity much larger to see if it improves

2. Try their model on the data and see what the predictions look like

3. Plot histogram of training data sequences with fraction of fixations


