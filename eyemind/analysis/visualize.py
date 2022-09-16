import matplotlib.pyplot as plt
import numpy as np


def plot_figures(preds, targets):
    num_plots = len(preds)
    rows = (num_plots // 5) + 1
    cols = 5
    plt.figure(0)
    for i in range(rows):
        for j in range(cols):
            ax = plt.subplot2grid((rows,cols), (i,j))
            ax.step(np.arange(len(preds[0])),targets[j * i].numpy(), preds[j * i].numpy())
    plt.show()

