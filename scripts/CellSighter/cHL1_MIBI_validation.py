import sys
sys.path.append(".")
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import argparse
import numpy as np
from model import Model
from data.data import CellCropsDataset
from data.utils import load_crops
from data.transform import train_transform, val_transform
from torch.utils.data import DataLoader, WeightedRandomSampler
import json
import pandas as pd

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

def results_reformatting(result_dir, class_names, fold_id):
    res_df_cols = ['%s_prob' % class_name for class_name in class_names]
    res_df_cols.append('pred_label')
    res_df_cols.append('gt_label')

    res = pd.read_csv(os.path.join(result_dir, fold_id,  'results.csv'))
    gt_labels = res['gt_labels'].to_numpy()
    pred_labels = res['pred_labels'].to_numpy()
    pred_probs = res['pred_probs'].tolist()
    probs = np.zeros((len(pred_probs), len(class_names)))
    for i in range(len(pred_probs)):
        probs[i, :] = [np.float64(value) for value in pred_probs[i][1:-1].split(',')]
    gt_labels = np.expand_dims(gt_labels, axis=1)
    pred_labels = np.expand_dims(pred_labels, axis=1)
    res_df = pd.DataFrame(np.concatenate((probs, pred_labels, gt_labels), axis=1), columns=res_df_cols)
    res_df.to_csv(os.path.join(result_dir, fold_id, 'results_test.csv'), index=False)


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
    parser.add_argument('--base_path', type=str, default='datasets/cHL1_MIBI/', help='configuration_path')
    parser.add_argument('--result_path', type=str, default='results/cHL1_MIBI/', help='configuration_path')
    parser.add_argument('--fold_id', type=str, default='fold_0', help='fold id to train model')
    args = parser.parse_args()

    fold_LUT = pd.read_csv('datasets/cHL1_MIBI/image_to_fold_mapping.csv')
    os.makedirs(os.path.join(args.result_path, args.fold_id), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.result_path, 'logs', args.fold_id))
    config_path = os.path.join(args.base_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    fid = args.fold_id.split('_')[-1]
    config["train_set"] = list(fold_LUT.loc[fold_LUT['fold_id'] != int(fid), 'folder_name'])
    config["train_set"].append('S354_1_rLN_ctrl_R1_tile5x5')
    config["train_set"].append('S354_2_rLN_ctrl_R2_tile5x5')
    config["test_set"] = list(fold_LUT.loc[fold_LUT['fold_id'] == int(fid), 'folder_name'])
    criterion = torch.nn.CrossEntropyLoss()
    _, test_crops = load_crops(config["root_dir"],
                                        config["channels_path"],
                                        config["crop_size"],
                                        [],
                                        config["test_set"],
                                        config["to_pad"],
                                        blacklist_channels=config["blacklist"])

    test_crops = np.array([c for c in test_crops if c._label >= 0])
    shift = 5
    crop_input_size = config["crop_input_size"] if "crop_input_size" in config else 100
    test_dataset = CellCropsDataset(test_crops, transform=val_transform(crop_input_size), mask=True)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=16, shuffle=False)
    print(len(test_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_channels = sum(1 for line in open(config["channels_path"])) + 1 - len(config["blacklist"])
    class_num = config["num_classes"]

    eval_weights = os.path.join(args.result_path, args.fold_id, "weights.pth")
    model = Model(num_channels + 1, class_num)
    model.load_state_dict(torch.load(eval_weights))
    model = model.to(device=device)

    pred_labels, pred_probs, results_df = test_epoch(model, test_loader, device=device)
    results_df.to_csv(os.path.join(args.result_path, args.fold_id, "results.csv"), index=False)
    class_names = pd.read_csv(os.path.join(args.result_path, 'class_names.csv'))['class_name'].tolist()
    results_reformatting(args.result_path, class_names, args.fold_id)
