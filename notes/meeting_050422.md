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