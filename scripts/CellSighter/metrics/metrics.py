import torch
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Metrics:
    """
    Class that aggregate all the results of the network and run set of metrics on the results.
    """
    def __init__(self, metrics, tb, prefix=""):
        self.metrics = metrics
        self.tb = tb
        self.prefix = prefix

    def attach(self, m):
        """
        Attack new metric to run of the data
        """
        self.metrics.append(m)

    def preprocess_cells(self, cells_crop):
        output = {}
        for batch in cells_crop:
            for k, v in batch.items():
                if k not in output:
                    output[k] = []
                if isinstance(v, torch.Tensor):
                    output[k] += v.cpu().numpy().tolist()
                else:
                    output[k] += v
        return output

    def __call__(self, cells_crop, pred_prob, epoch):
        cells_crop = self.preprocess_cells(cells_crop)
        for m in self.metrics:
            np.save(f"{m.__class__.__name__}", m(cells_crop, pred_prob, self.tb, epoch, self.prefix))

    def save_results(self, path, cells_crop, pred_prob):
        cells_crop = self.preprocess_cells(cells_crop)
        cells_crop = pd.DataFrame(cells_crop)
        cells_crop["pred"] = pred_prob.argmax(1)
        cells_crop["pred_prob"] = pred_prob.max(1)
        cells_crop["prob_list"] = pred_prob.tolist()
        cells_crop[["pred", "pred_prob", "label", "cell_id", "image_id", "prob_list"]].to_csv(path)

    def get_results(self, cells_crop, pred_prob):
        cells_crop = self.preprocess_cells(cells_crop)
        cells_crop = pd.DataFrame(cells_crop)
        cells_crop["pred"] = pred_prob.argmax(1)
        cells_crop["pred_prob"] = pred_prob.max(1)
        cells_crop["prob_list"] = pred_prob.tolist()
        cells_crop[["pred", "pred_prob", "label", "cell_id", "image_id", "prob_list"]].to_csv(path)
class ConfusionMatrixMetric:
    def __init__(self, mapping=None, labels_name=None, name=None):
        self.mapping = mapping
        self.labels_name = labels_name
        self.name = name

    def __call__(self, cells, pred_prob, tb, epoch, prefix):
        gt_labels = cells["label"]
        pred = pred_prob.argmax(1)
        cm = confusion_matrix(gt_labels, pred, normalize="true")
        fig, ax = plt.subplots(figsize=(15, 15))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)

        tb.add_figure(f"{prefix}_{self.name}_cm", fig, epoch)
        return cm