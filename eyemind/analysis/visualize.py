import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import torch
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

def plot_scanpath_labels_df(df,event=None,exclude=None, label=None):
    if event:
        temp_df = df.loc[df['event']==event]
    else: 
        temp_df=df
    if exclude: # filter out flag vals
        temp_df=temp_df[(temp_df["XAvg"]!=exclude) & (temp_df["YAvg"]!=exclude)]
    plt.plot(temp_df["XAvg"], -temp_df["YAvg"], color='k')
    if label:
        groups = temp_df.groupby(label)
        # colors = cm.rainbow(np.linspace(0, 1, len(ys)))
        for name, group in groups:
            plt.scatter(group.XAvg, -group.YAvg, s=(15 if name>0 else 1), alpha=.4, label=name)
      # for y, c in zip(ys, colors):
      #     plt.scatter(x, y, color=c)
        plt.legend(title=label)
    name = f'{temp_df["ParticipantID"].iloc[0]} {temp_df["event"].iloc[0]}'
    plt.title(name)
    plt.show() 

def plot_scanpath_labels(x,y, labels=None, remove_masked=True):
    na_mask_val=-180
    # remove masked values
    if remove_masked:
        x[x<=na_mask_val]=np.nan
        y[y<=na_mask_val]=np.nan
        if labels is not None:
            labels[labels==na_mask_val]=np.nan
    fig,ax = plt.subplots()
    ax.plot(x, -y, color='k')
    if labels:
        grouped = zip(x,y,labels)
        groups = {}
        unique_labels = set(labels)
        for label in unique_labels:
            ax.scatter(x[labels==label], -y[labels==label], s=(15 if label>0 else 1), alpha=.4, label=label)
        ax.legend(title=label)
    return fig

def plot_scanpath_pc(x,y,pred_x,pred_y, remove_masked=True):
    na_mask_val=-180
    # remove masked values
    if remove_masked:
        x[x<=na_mask_val]=np.nan
        y[y<=na_mask_val]=np.nan
    fig,ax = plt.subplots()
    ax.plot(x, -y, color='k')
    ax.plot(pred_x, -pred_y, color='r')
    return fig

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


def fixation_image(pred,target, title="Fixation Identification (top:pred, bottom: target)"):
    sl = len(pred)
    print(f'sequence length: {sl}')
    fixation_labels = torch.cat((pred.expand(sl//2,sl),target.expand(sl//2,sl)))
    plt.imshow(fixation_labels, extent=[0, len(fixation_labels[1]),0, 100], cmap='Greys')
    # plt.xticks(np.arange(0, len(fixation_labels), 1), [])
    plt.yticks([])
    # plt.grid(True, axis='x', lw=1, c='black')
    # plt.tick_params(axis='x', length=0)
    plt.title(title)
    black = mpatches.Patch(color='black', label='Fixation')
    white = mpatches.Patch(color='white', label='Saccade')
    plt.legend(handles=[black, white],bbox_to_anchor=(1.15, 1), loc='upper right')
    plt.xlabel("Time Steps")
    plt.show()
