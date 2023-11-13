import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

FILE_FORMAT='SVG'
FILE_EXT = FILE_FORMAT.lower()

def conf_matrix(res_dir, fold_ids, class_names, class_ids, result_csv_name='results_valid.csv'):
    
    gt = []
    pred = []
    for fid in fold_ids:
        exp_path = '%s/fold_%d/%s' % (res_dir, fid, result_csv_name)
        res_data = pd.read_csv(exp_path)
        if len(gt) == 0:
            gt = res_data['gt_label'].tolist()
            pred = res_data['pred_label'].tolist()
        else:
            gt.extend(res_data['gt_label'].tolist())
            pred.extend(res_data['pred_label'].tolist())
    
    gt_ = np.array(gt)
    pred_ = np.array(pred)
    gt = np.zeros((gt_.shape[0],))
    pred = np.zeros((pred_.shape[0],))
    for i in range(len(class_ids)):
        gt[gt_==class_ids[i]] = i
        pred[pred_==class_ids[i]] = i
    
    conf = np.zeros((len(class_names), len(class_names)))
    conf = confusion_matrix(gt, pred, labels=[i for i in range(len(class_names))])
    
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(1,1,1)
#     plot_confusion_matrix(conf_mat=conf, figsize=(12, 12), show_absolute=False, show_normed=True, class_names=class_names, axis=ax)
#     fig.savefig(os.path.join(res_dir, 'conf_matrix_percentage.%s' % FILE_EXT), format=FILE_FORMAT, bbox_inches='tight')
    
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(1,1,1)
#     plot_confusion_matrix(conf_mat=conf, figsize=(12, 12), show_absolute=True, show_normed=False, class_names=class_names, axis=ax)
#     fig.savefig(os.path.join(res_dir, 'conf_matrix_count.%s' % FILE_EXT), format=FILE_FORMAT, bbox_inches='tight')
    
    fig = plt.figure(figsize=(10, 10), dpi=600)
    ax = fig.add_subplot(1,1,1)
    plot_confusion_matrix(conf_mat=conf, figsize=(10, 10), show_normed=True, show_absolute=True, class_names=class_names, axis=ax)
    fig.savefig(os.path.join(res_dir, 'conf_matrix.%s' % FILE_EXT), format=FILE_FORMAT, bbox_inches='tight')
    
def accuracy_plots(res_dir, fold_ids, class_names, class_ids, color_palette, result_csv_name='results_valid.csv'):
    class_count = len(class_names)
    group_name = []
    accuracy = []

    res = np.zeros((len(fold_ids), len(class_names)+1))
    for fid in fold_ids:
        exp_path = '%s/fold_%d/%s' % (res_dir, fid, result_csv_name)
        res_data = pd.read_csv(exp_path)
        gt_ = np.array(res_data['gt_label'].tolist())
        pred_ = np.array(res_data['pred_label'].tolist())
        gt = np.zeros((gt_.shape[0],))
        pred = np.zeros((pred_.shape[0],))
        for i in range(len(class_ids)):
            gt[gt_==class_ids[i]] = i
            pred[pred_==class_ids[i]] = i
        
        acc = []
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
            
            acc.append(f1)
            res[fid, class_id] = acc[-1]
        acc_ = np.sum(gt==pred)/gt_.shape[0]
        prec = tp_all/(tp_all+fp_all)
        recall = tp_all/(tp_all+fn_all)
        f1 = (2*prec*recall)/(prec+recall)
        acc.append(f1)
        res[fid, class_id+1] = acc[-1]
        accuracy.append(acc)
        group_name.append('fold-%d' % (fid+1))
        print('Fold-%d: Accuracy: %.4f, Precision: %0.4f, Recall: %0.4f, F1-Score: %0.4f' % (fid+1, acc_, prec, recall, f1))
    
    class_names.append('Average') 
    fig = plt.figure(figsize=(10, 2), dpi=600)
    ax = fig.add_subplot(1,1,1)
    x = np.arange(len(class_names))

    offset = [-0.30, -0.15, 0, 0.15, 0.30 ]
    width = 0.1

    for n in range(len(group_name)):
        ax.bar(x + offset[n], accuracy[n], width, label=group_name[n], color=color_palette[n])
    ax.set_ylabel('F1-Score')
    ax.set_ylim([0.0, 1.0])
    ax.set_yticks([0.20, 0.40, 0.60, 0.8, 1.0])
    ax.set_xticks(x, class_names, rotation=90)
    ax.legend(loc='lower left', ncol=5)
    
    fig.savefig(os.path.join(res_dir, 'cell_level_f1_score_plot.%s' % FILE_EXT), format=FILE_FORMAT, bbox_inches='tight')
    
    df = pd.DataFrame(res, columns=class_names, index=[i for i in range(5)])
    df.to_csv(os.path.join(res_dir, 'cell_level_f1_score_plot.csv'))
    

def performance_plots(res_dir, fold_ids, class_names, class_ids, color_palette, result_csv_name='results_valid.csv'):
    class_count = len(class_names)
    group_name = []
    res = np.zeros((len(fold_ids), len(class_names)+1, 3))
    for fid in fold_ids:
        exp_path = '%s/fold_%d/%s' % (res_dir, fid, result_csv_name)
        res_data = pd.read_csv(exp_path)
        gt_ = np.array(res_data['gt_label'].tolist())
        pred_ = np.array(res_data['pred_label'].tolist())
        gt = np.zeros((gt_.shape[0],))
        pred = np.zeros((pred_.shape[0],))
        for i in range(len(class_ids)):
            gt[gt_==class_ids[i]] = i
            pred[pred_==class_ids[i]] = i
        
        for class_id in range(class_count):
            T = gt==(class_id)
            P = pred==(class_id)
            tp = np.sum(gt[T]==pred[T])
            fp = np.sum(gt[P]!=pred[P])
            fn = np.sum(gt[T]!=pred[T])

            prec = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1 = (2*prec*recall)/(prec+recall)
   
            res[fid, class_id, 0] = prec
            res[fid, class_id, 1] = recall
            res[fid, class_id, 2] = f1
        res[np.isnan(res)] = 0
        prec = np.mean(res[fid, :-1, 0])
        recall = np.mean(res[fid,:-1, 1])
        f1 = np.mean(res[fid, :-1, 2])
        res[fid, class_id+1, 0] = prec
        res[fid, class_id+1, 1] = recall
        res[fid, class_id+1, 2] = f1
    
    mean_res = np.mean(res, axis=0)
    std_res = np.std(res, axis=0)
    
    class_names.append('Average') 
    group_name = ['Precision', 'Recall', 'F1-Score']
    fig = plt.figure(figsize=(10, 2), dpi=600)
    ax = fig.add_subplot(1,1,1)
    x = np.arange(len(class_names))

    offset = [-0.20, 0, 0.20]
    width = 0.15

    for n in range(len(group_name)):
        ax.bar(x + offset[n], mean_res[:,n].tolist(), width, yerr=std_res[:,n].tolist(), label=group_name[n], color=color_palette[n+1])
    ax.set_ylabel('Score')
    ax.set_ylim([0.0, 1.0])
    ax.set_yticks([0.0, 0.20, 0.40, 0.60, 0.80, 1.0])
    ax.set_xticks(x, class_names, rotation=90)
    ax.legend(loc='lower left', ncol=3)
    
    fig.savefig(os.path.join(res_dir, 'cell_level_performance_plot.%s' % FILE_EXT), format=FILE_FORMAT, bbox_inches='tight')
    data_dict = {'class_names': class_names, 'means': mean_res, 'stds': std_res, 'group_name': group_name}
    np.save(os.path.join(res_dir, 'cell_level_performance_plot.npy'), data_dict)
    pd.DataFrame(res[:, :, 0], columns=class_names, index=[i for i in range(5)]).to_csv(os.path.join(res_dir, 'cell_level_precision.csv'))
    pd.DataFrame(res[:, :, 1], columns=class_names, index=[i for i in range(5)]).to_csv(os.path.join(res_dir, 'cell_level_recall.csv'))
    pd.DataFrame(res[:, :, 2], columns=class_names, index=[i for i in range(5)]).to_csv(os.path.join(res_dir, 'cell_level_f1_score.csv'))
    
def micro_average_pr_curve(res_dir, fold_ids, class_names, class_ids, result_csv_name='results_valid.csv'):
    fig = plt.figure(figsize=(6, 6), dpi=600)
    ax = fig.add_subplot(1,1,1)
    for fid in fold_ids:
        exp_path = '%s/fold_%d/%s' % (res_dir, fid, result_csv_name)
        res_data = pd.read_csv(exp_path)
        gt_ = np.array(res_data['gt_label'].tolist())
        pred_ = np.array(res_data['pred_label'].tolist())
        gt = np.zeros((gt_.shape[0],))
        pred = np.zeros((pred_.shape[0],))
        for i in range(len(class_ids)):
            gt[gt_==class_ids[i]] = i
            pred[pred_==class_ids[i]] = i
            
        pred_probs_ = res_data.to_numpy()[:,:-2]
        pred_probs = np.zeros((pred_probs_.shape[0], pred_probs_.shape[1]))
        for i in range(len(class_ids)):
            pred_probs[:,i] = pred_probs_[:,class_ids[i]]


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

        display = PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"],
        )
        display.plot(ax=ax, name='Fold %d' % fid)
    _ = display.ax_.set_title("Micro-averaged over all classes")
    fig.savefig(os.path.join(res_dir, 'micro_average_pr_curve.%s' % FILE_EXT), format=FILE_FORMAT, bbox_inches='tight')

def average_pr_curve_per_class(res_dir, fold_ids, class_names, class_ids, color_palette, result_csv_name='results_valid.csv'):
    gt_ = None
    pred_ = None
    pred_probs_ = None
    for fid in fold_ids:
        exp_path = '%s/fold_%d/%s' % (res_dir, fid, result_csv_name)
        res_data = pd.read_csv(exp_path)
        if gt_ is None:
            gt_ = np.array(res_data['gt_label'].tolist())
            pred_ = np.array(res_data['pred_label'].tolist())
            pred_probs_ = res_data.to_numpy()[:,:-2]
        else:
            gt_ = np.concatenate((gt_, np.array(res_data['gt_label'].tolist())), axis=0)
            pred_ = np.concatenate((pred_, np.array(res_data['pred_label'].tolist())), axis=0)
            pred_probs_ = np.concatenate((pred_probs_, res_data.to_numpy()[:,:-2]), axis=0)
    
    gt = np.zeros((gt_.shape[0],))
    pred = np.zeros((pred_.shape[0],))
    pred_probs = np.zeros((pred_probs_.shape[0], pred_probs_.shape[1]))
    for i in range(len(class_ids)):
        gt[gt_==class_ids[i]] = i
        pred[pred_==class_ids[i]] = i
        pred_probs[:,i] = pred_probs_[:,class_ids[i]]
    
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
    # colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
    colors = color_palette

    fig, ax = plt.subplots(figsize=(6, 6), dpi=600)

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
    
    fig.savefig(os.path.join(res_dir, 'average_pr_curve_per_class.%s' % (FILE_EXT)), format=FILE_FORMAT, bbox_inches='tight')