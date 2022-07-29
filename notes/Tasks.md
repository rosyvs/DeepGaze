## 7/17/22

### Multi-task training:

1. Weighting of different tasks: In OBF they use an equal weighting for each task but the two different types of loss functions, RMSE and Cross Entropy, have different distributions of values which could effect the weighting of the tasks. 

Solutions:

- Somehow weigh the tasks so that the parameters are influenced more equally
- Hyperparameter search for best weighting or find a method that works well for this

2. Dealing with Flag values for predictive coding and reconstruction: the flag value -180 will have a large influence over the values predicted by the model since it will have a large loss more often

Solutions:

- Set a max value for the loss

3. Contrastive learning

- Troubleshoot results as the model is achieving almost 100% accuracy
- Explore other types of self-supervised representation learning that don't exhibit mode collapse: BYOL, SIMCLR, Triplet Loss

4. Check learning rates

### Informer Model

1. Normalize data as the paper does
2. Test different hyperparameters


### Visualizing Outputs

1. Use RMSE on y-axis for predictive coding and reconstruction to get the visual degrees back
2. Plot all tasks on one plot
3. How can I adjust axis scale easily in tensorboard?



