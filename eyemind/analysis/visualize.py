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

def viz_coding(inputs, preds, title):
  """
  
  Args:
    inputs (torch.tensor): the input data s.t. sequence with shape 
      (batch_size, whole_sequence_length, 2)
    preds (torch.tensor): predicted following sequence with shape
      (batch_size, pred_sequence_length, 2)
    title (str): visualization title
    
  """

  n1 = inputs.shape[1]
  n2 = preds.shape[1]
  plt.figure(dpi=1200)
  # The input sequence
  plt.plot(list(range(n1)),
           inputs[0, :, 0].detach().cpu().numpy(),
           label="x",
           color="orange")
  plt.plot(list(range(n1)),
           inputs[0, :, 1].detach().cpu().numpy(),
           label="y",
           color="blue")
  plt.ylim([-10, 10])

  # The pred sequence
  plt.plot(list(range(n1-n2, n1)),
              preds[0, :, 0].detach().cpu().numpy(),
              label="x pred",
              color="red")
  plt.plot(list(range(n1-n2, n1)),
              preds[0, :, 1].detach().cpu().numpy(),
              label="y pred",
              color="green")

  plt.title(title)
  plt.xlabel("Time Steps (500 steps ~ 8.5s)")
  plt.ylabel("Visual Angle (screen center is (0,0))")
  plt.legend()
  plt.show()
