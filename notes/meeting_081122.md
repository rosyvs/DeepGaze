## Updates

1. Ran transformer model on fixation ID with normalization for 50 epochs. Not seeing good results. Trying to run with predictive coding since that is what their model was tested on.
2. Set max loss and changed data loading so that it is random parts of the sequence. Fixed dataloading for the contrastive learning task 
3. Waiting for tests to run but in queue
4. Currently checking to make sure mask is correct for fixation identification and refactoring some dataloading functions

## Notes

- Adding velocity to the input data
- Regression data 
- If showing result show the number of samples that go into the result
- Measurement of whether it is predicting in blocks of fixations and saccades or whether it is more random, smoothness to loss function