import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from itertools import cycle
from mlxtend.plotting import plot_confusion_matrix

def visulize_training_logs(training_log_csv):
    data = pd.read_csv(training_log_csv)
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(1,3,1)
    epochs = range(1,data.shape[0]+1)
    ax.plot(epochs, data['train_loss'].to_numpy(), 'g', label='Training Loss')
    ax.plot(epochs, data['valid_loss'].to_numpy(), 'b', label='Validation Loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    ax = fig.add_subplot(1,3,2)
    epochs = range(1,data.shape[0]+1)
    ax.plot(epochs, data['train_acc'].to_numpy(), 'g', label='Training Accuracy')
    ax.plot(epochs, data['valid_acc'].to_numpy(), 'b', label='Validation Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()

    ax = fig.add_subplot(1,3,3)
    epochs = range(1,data.shape[0]+1)
    ax.plot(epochs, data['train_auc'].to_numpy(), 'g', label='Training AUC')
    ax.plot(epochs, data['valid_auc'].to_numpy(), 'b', label='Validation AUC')
    ax.set_title('Training and Validation AUC')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    plt.show()

def conf_matrix(gt, pred, class_names):
    conf = np.zeros((len(class_names), len(class_names)))
    conf = confusion_matrix(gt, pred, labels=[i for i in range(len(class_names))])
    
    fig = plt.figure(figsize=(10, 10), dpi=600)
    ax = fig.add_subplot(1,1,1)
    plot_confusion_matrix(conf_mat=conf, figsize=(12, 12), show_absolute=True, show_normed=True, class_names=class_names, axis=ax)
    
def pre_recall_f1_score(gt, pred, class_names, color_palette):
    class_count = len(class_names)
    res = np.zeros((len(class_names)+1,3))
    tp_all = 0
    fp_all = 0
    fn_all = 0
    for class_id in range(class_count):
        T = gt==(class_id)
        P = pred==(class_id)
        tp = np.sum(gt[T]==pred[T])
        fp = np.sum(gt[P]!=pred[P])
        fn = np.sum(gt[T]!=pred[T])

        tp_all += tp
        fp_all += fp
        fn_all += fn

        prec = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = (2*prec*recall)/(prec+recall)

        res[class_id, 0] = prec
        res[class_id, 1] = recall
        res[class_id, 2] = f1

    prec = tp_all/(tp_all+fp_all)
    recall = tp_all/(tp_all+fn_all)
    f1 = (2*prec*recall)/(prec+recall)
    res[class_id+1, 0] = prec
    res[class_id+1, 1] = recall
    res[class_id+1, 2] = f1
       
    class_names.append('Average') 
    group_name = ['Precision', 'Recall', 'F1-Score']
    fig = plt.figure(figsize=(10, 5), dpi=600)
    ax = fig.add_subplot(1,1,1)
    x = np.arange(len(class_names))

    offset = [-0.20, 0, 0.20]
    width = 0.15

    for n in range(len(group_name)):
        ax.bar(x+offset[n], res[:,n].tolist(), width, label=group_name[n], color=color_palette[n])
    ax.set_ylabel('Score')
    ax.set_ylim([0.20, 1.0])
    ax.set_yticks([0.20, 0.40, 0.60, 0.80, 1.0])
    ax.set_xticks(x, class_names, rotation=90)
    ax.legend(loc='lower left')
    
def average_pr_curve_per_class(gt, pred, pred_probs, class_names):
        
    # Use label_binarize to be multi-label like settings
    Y_test = label_binarize(gt, classes=[i for i in range(len(class_names))])
    n_classes = Y_test.shape[1]
    y_score = pred_probs

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        Y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")

    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    fig, ax = plt.subplots(figsize=(10, 10), dpi=600)

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average", color="gold")

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"{class_names[i]}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="lower left")
    ax.set_title("Precision-Recall Curves")

    plt.show()
