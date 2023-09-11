import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap


def metric(gt, pred, classes_for_cm, colorbar=True):
    sns.set(font_scale=2)
    cm_normed_recall = confusion_matrix(gt, pred, labels=classes_for_cm, normalize="true") * 100
    cm = confusion_matrix(gt, pred, labels=classes_for_cm)

    plt.figure(figsize=(50,45))
    ax1 = plt.subplot2grid((50,50), (0,0), colspan=30, rowspan=30)
    cmap = LinearSegmentedColormap.from_list('', ['white', *plt.cm.Blues(np.arange(255))])
    annot_labels = cm_normed_recall.round(1).astype(str)
    annot_labels = pd.DataFrame(annot_labels) + "\n (" + pd.DataFrame(cm).astype(str)+")"

    annot_mask = cm_normed_recall.round(1) <= 0.1
    annot_labels[annot_mask] = ""

    sns.heatmap(cm_normed_recall.T, ax=ax1, annot=annot_labels.T, fmt='',cbar = colorbar,
                cmap=cmap,linewidths=1, vmin=0, vmax=100,linecolor='black', square=True)

    ax1.xaxis.tick_top()
    ax1.set_xticklabels(classes_for_cm,rotation=90)
    ax1.set_yticklabels(classes_for_cm,rotation=0)
    ax1.tick_params(axis='both', which='major', labelsize=35)

    ax1.set_xlabel("Clustering and gating", fontsize=35)
    ax1.set_ylabel("CellSighter", fontsize=35)


results = pd.read_csv(r"") #Fill in the path to your results file
classes_for_cm = np.unique(np.concatenate([results["label"], results["pred"]]))
metric(results["label"], results["pred"], classes_for_cm)
plt.savefig("confusion_matrix.png")