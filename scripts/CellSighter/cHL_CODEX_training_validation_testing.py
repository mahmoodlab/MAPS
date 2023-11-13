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
from sklearn.model_selection import StratifiedKFold
torch.multiprocessing.set_sharing_strategy('file_system')


def train_epoch(model, dataloader, optimizer, criterion, epoch, writer, device=None):
    model.train()
    cells = []
    for i, batch in enumerate(dataloader):
        x = batch['image']
        m = batch.get('mask', None)
        if m is not None:
            x = torch.cat([x, m], dim=1)
        x = x.to(device=device)

        y = batch['label'].to(device=device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        if i % 1 == 0:
            print(f"train epoch {epoch} | iterate {i} / {len(dataloader)} | {loss.item()}                         ", end='\r')
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)
        loss.backward()
        optimizer.step()
    return cells
def valid_epoch(model, dataloader, criterion, epoch, writer, device=None):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x = batch['image']
            m = batch.get('mask', None)
            if m is not None:
                x = torch.cat([x, m], dim=1)
            x = x.to(device=device)

            y = batch['label'].to(device=device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            if i % 1 == 0:
                print(f"valid epoch {epoch} | iterate {i} / {len(dataloader)} | {loss.item()}                         ", end='\r')
            writer.add_scalar('Loss/valid', loss.item(), epoch * len(dataloader) + i)
        total_loss = total_loss/i
    return total_loss

def val_epoch(model, dataloader, device=None):
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
    class_df = pd.read_csv(os.path.join(result_dir, 'class_names.csv'))
    class_names = {id: name for id, name in enumerate(class_df['class_name'].tolist())}
    col_names = ["%s_prob" % class_names[key] for key in class_names.keys()]
    class_names = [class_names[key] for key in class_names.keys()]

    res = pd.read_csv(os.path.join(result_dir, fold_id, 'test_results.csv'))

    gt_labels = res['gt_labels'].tolist()
    pred_labels = res['pred_labels'].tolist()
    probs = res['pred_probs'].to_numpy()
    prob_matrix = np.zeros((probs.shape[0], len(class_names)))
    for i in range(probs.shape[0]):
        class_prob = np.array([np.float64(val) for val in probs[i][1:-1].split(',')])
        prob_matrix[i, :] += class_prob

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

# CUDA_VISIBLE_DEVICES=1 python cHL_CODEX_training_validation_testing.py --fold_id fold_1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--base_path', type=str, default='datasets/cHL_CODEX/', help='configuration_path')
    parser.add_argument('--result_path', type=str, default='results/cHL_CODEX/', help='configuration_path')
    parser.add_argument('--fold_id', type=str, default='fold_0', help='fold id to train model')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.result_path, args.fold_id), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.result_path, 'logs', args.fold_id))
    config_path = os.path.join(args.base_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    criterion = torch.nn.CrossEntropyLoss()
    train_crops, val_crops = load_crops(config["root_dir"],
                                        config["channels_path"],
                                        config["crop_size"],
                                        config["train_set"],
                                        config["val_set"],
                                        config["to_pad"],
                                        blacklist_channels=config["blacklist"])

    train_crops = np.array([c for c in train_crops if c._label >= 0])
    val_crops = np.array([c for c in val_crops if c._label >= 0])

    # creating data folds during run time
    labels = np.array([c._label for c in train_crops if c._label >= 0])
    skf_train_test = StratifiedKFold(n_splits=5, random_state=7325111, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf_train_test.split(labels, labels)):
        if ("fold_%d" % i) == args.fold_id:
            print(f"Fold {i}, {len(train_index)}, {len(test_index)}:")
            test_crops = train_crops[test_index]
            train_crops = train_crops[train_index]
            labels = np.array([c._label for c in train_crops])
            skf_train_valid = StratifiedKFold(n_splits=5, random_state=7325111, shuffle=True)
            for j, (train_index, valid_index) in enumerate(skf_train_valid.split(labels, labels)):
                    print(f"Fold {j}, {len(train_index)}, {len(test_index)}:")
                    val_crops = train_crops[valid_index]
                    train_crops = train_crops[train_index]
                    break
            break

    sampler = define_sampler(train_crops)
    shift = 5
    crop_input_size = config["crop_input_size"] if "crop_input_size" in config else 100
    aug = config["aug"] if "aug" in config else True
    training_transform = train_transform(crop_input_size, shift) if aug else val_transform(crop_input_size)
    train_dataset = CellCropsDataset(train_crops, transform=training_transform, mask=True)
    val_dataset = CellCropsDataset(val_crops, transform=val_transform(crop_input_size), mask=True)
    test_dataset = CellCropsDataset(test_crops, transform=val_transform(crop_input_size), mask=True)
    train_dataset_for_eval = CellCropsDataset(train_crops, transform=val_transform(crop_input_size), mask=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_channels = sum(1 for line in open(config["channels_path"])) + 1 - len(config["blacklist"])
    class_num = config["num_classes"]

    model = Model(num_channels + 1, class_num)

    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=12, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=12, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=12, shuffle=False)
    print(len(train_loader), len(val_loader))
    min_loss = np.Inf
    best_epoch_id = -1
    patient = 0
    for i in range(config["epoch_max"]):
        train_epoch(model, train_loader, optimizer, criterion, device=device, epoch=i, writer=writer)
        total_loss = valid_epoch(model, val_loader, criterion, device=device, epoch=i, writer=writer)
        print(f"Epoch {i} done with valid loss: {total_loss}!")
        patient += 1

        if total_loss < min_loss:
            patient = 0
            min_loss = total_loss
            best_epoch_id = i
            print(f"Saving new best model at epoch {i} with valid loss: {total_loss}!")
            torch.save(model.state_dict(), os.path.join(args.result_path, args.fold_id, "weights.pth"))
            pred_labels, pred_probs, results_df = val_epoch(model, test_loader, device=device)
            results_df.to_csv(os.path.join(args.result_path, args.fold_id, "test_results.csv"), index=False)
            results_reformatting(args.result_path, args.fold_id)

        if i > 50 and patient >= 20:
            print('Loss has not decreased in last 20 epochs. Therefore, terminating training process.')
            break