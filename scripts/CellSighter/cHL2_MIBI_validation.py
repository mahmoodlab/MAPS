import sys
sys.path.append(".")
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import argparse
import numpy as np
import pandas as pd
from model import Model
from data.data import CellCropsDataset
from data.utils import load_crops
from data.transform import train_transform, val_transform
from torch.utils.data import DataLoader, WeightedRandomSampler
import json

torch.multiprocessing.set_sharing_strategy('file_system')


def test_epoch(model, dataloader, device=None):
    with torch.no_grad():
        model.eval()
        pred_labels = []
        pred_probs = []
        results = {'cell_id': [], 'image_id': [], 'gt_labels': [], 'pred_labels': [], 'pred_probs': []}

        for i, batch in enumerate(dataloader):
            x = batch['image']
            m = batch.get('mask', None)
            if m is not None:
                x = torch.cat([x, m], dim=1)
            x = x.to(device=device)
            # m = m.to(device=device)
            y_pred = model(x)

            pred_probs += y_pred.detach().cpu().numpy().tolist()
            pred_labels += y_pred.detach().cpu().numpy().argmax(1).tolist()

            results['cell_id'].extend(batch['cell_id'].detach().cpu().numpy().tolist())
            results['image_id'].extend(batch['image_id'])
            results['gt_labels'].extend(batch['label'].detach().cpu().numpy().tolist())
            results['pred_labels'].extend(np.argmax(y_pred.detach().cpu().numpy(), axis=1).tolist())
            results['pred_probs'].extend(y_pred.detach().cpu().numpy().tolist())

            print(f"Eval {i} / {len(dataloader)}        ", end='\r')
        return np.array(pred_labels), np.array(pred_probs), pd.DataFrame.from_dict(results)

def results_reformatting(result_dir, fold_id):
    class_ids = {"0": "0", "1": "1", "2": "1", "3": "2", "4": "3", "5": "4", "6": "5", "7": "6", "8": "7", "9": "9",
                 "10": "8", "11": "10", "12": "1", "13": "11"}
    class_names = {"0": "B", "1": "CD4 T", "2": "CD4 Treg", "3": "CD8 T", "4": "DC", "5": "Endothelial", "6": "M1",
                   "7": "M2", "8": "NK", "9": "Neutrophil", "10": "Other", "11": "Tumor"}
    col_names = ["%s_prob" % class_names[key] for key in class_names.keys()]

    res = pd.read_csv(os.path.join(result_dir, fold_id, 'results.csv'))

    gt_labels = [int(class_ids[str(label)]) for label in res['gt_labels'].tolist()]
    pred_labels = [int(class_ids[str(label)]) for label in res['pred_labels'].tolist()]
    probs = res['pred_probs'].to_numpy()
    prob_matrix = np.zeros((probs.shape[0], len(col_names)))
    for i in range(probs.shape[0]):
        class_prob = np.array([np.float64(val) for val in probs[i][1:-1].split(',')])
        for key in class_ids.keys():
            prob_matrix[i, int(class_ids[key])] += class_prob[int(class_ids[key])]

    df = pd.DataFrame(prob_matrix, columns=col_names)
    df['pred_label'] = pred_labels
    df['gt_label'] = gt_labels
    os.makedirs(os.path.join(result_dir, fold_id), exist_ok=True)
    df.to_csv(os.path.join(result_dir, fold_id, 'results_test.csv'), index=False)

def subsample_const_size(crops, size):
    """
    sample same number of cell from each class
    """
    final_crops = []
    crops = np.array(crops)
    labels = np.array([c._label for c in crops])
    for lbl in np.unique(labels):
        indices = np.argwhere(labels == lbl).flatten()
        if (labels == lbl).sum() < size:
            chosen_indices = indices
        else:
            chosen_indices = np.random.choice(indices, size, replace=False)
        final_crops += crops[chosen_indices].tolist()
    return final_crops


def define_sampler(crops, hierarchy_match=None):
    """
    Sampler that sample from each cell category equally
    The hierarchy_match defines the cell category for each class.
    if None then each class will be category of it's own.
    """
    labels = np.array([c._label for c in crops])
    if hierarchy_match is not None:
        labels = np.array([hierarchy_match[str(l)] for l in labels])

    unique_labels = np.unique(labels)
    class_sample_count = {t: len(np.where(labels == t)[0]) for t in unique_labels}
    weight = {k: sum(class_sample_count.values()) / v for k, v in class_sample_count.items()}
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)
    return WeightedRandomSampler(samples_weight.double(), len(samples_weight))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--base_path', type=str, default='datasets/cHL2_MIBI/', help='configuration_path')
    parser.add_argument('--result_path', type=str, default='results/cHL2_MIBI/', help='configuration_path')
    parser.add_argument('--fold_id', type=str, default='fold_0', help='fold id to train model')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.result_path, args.fold_id), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.result_path, 'logs', args.fold_id))

    config_path = os.path.join(args.base_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    num_channels = sum(1 for line in open(config["channels_path"])) + 1 - len(config["blacklist"])
    class_num = config["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(num_channels + 1, class_num)
    eval_weights = os.path.join(args.result_path, args.fold_id, 'weights.pth')
    model.load_state_dict(torch.load(eval_weights))
    model = model.to(device=device)
    model.eval()

    config["train_set"] = config["%s_train_set" % args.fold_id]
    config["test_set"] = config["%s_test_set" % args.fold_id]

    _, test_crops = load_crops(config["root_dir"],
                              config["channels_path"],
                              config["crop_size"],
                              [],
                              config["test_set"],
                              config["to_pad"],
                              blacklist_channels=config["blacklist"])

    test_crops = np.array([c for c in test_crops if c._label >= 0])
    crop_input_size = config["crop_input_size"] if "crop_input_size" in config else 100
    test_dataset = CellCropsDataset(test_crops, transform=val_transform(crop_input_size), mask=True)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=10, shuffle=False)
    print(len(test_loader))

    pred_labels, pred_probs, results_df = test_epoch(model, test_loader, device=device)
    results_df.to_csv(os.path.join(args.result_path, args.fold_id, f"results.csv"), index=False)
    results_reformatting(args.result_path, args.fold_id)


