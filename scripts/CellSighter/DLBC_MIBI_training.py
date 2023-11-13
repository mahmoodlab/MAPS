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
    parser.add_argument('--base_path', type=str, default='datasets/DLBC_MIBI/', help='configuration_path')
    parser.add_argument('--result_path', type=str, default='results/DLBC_MIBI/', help='configuration_path')
    parser.add_argument('--fold_id', type=str, default='fold_0', help='fold id to train model')
    args = parser.parse_args()

    fold_mapping = pd.read_csv('datasets/DLBC_MIBI/fov_to_fold_mapping.csv')
    ctrl_fovs = pd.read_csv('datasets/DLBC_MIBI/ctrl_fov.csv')['frame_name'].tolist()


    os.makedirs(os.path.join(args.result_path, args.fold_id), exist_ok=True)

    config_path = os.path.join(args.base_path, "config.json")
    writer = SummaryWriter(log_dir=os.path.join(args.result_path, 'logs', args.fold_id))
    with open(config_path) as f:
        config = json.load(f)
    fid = args.fold_id.split('_')[-1]
    config["train_set"] = list(fold_mapping.loc[fold_mapping['fold_id'] != int(fid), 'frame_name'])
    config["train_set"].extend(ctrl_fovs)

    criterion = torch.nn.CrossEntropyLoss()
    train_crops, _ = load_crops(config["root_dir"],
                                config["channels_path"],
                                config["crop_size"],
                                config["train_set"],
                                [],
                                config["to_pad"],
                                blacklist_channels=config["blacklist"])

    train_crops = np.array([c for c in train_crops if c._label >= 0])

    # creating data folds during run time
    labels = np.array([c._label for c in train_crops if c._label >= 0])
    skf_train_valid = StratifiedKFold(n_splits=5, random_state=7325111, shuffle=True)
    for j, (train_index, valid_index) in enumerate(skf_train_valid.split(labels, labels)):
            print(f"Fold {j}, {len(train_index)}, {len(valid_index)}:")
            val_crops = train_crops[valid_index]
            train_crops = train_crops[train_index]
            break

    sampler = define_sampler(train_crops)
    shift = 5
    crop_input_size = config["crop_input_size"] if "crop_input_size" in config else 100
    aug = config["aug"] if "aug" in config else True
    training_transform = train_transform(crop_input_size, shift) if aug else val_transform(crop_input_size)
    train_dataset = CellCropsDataset(train_crops, transform=training_transform, mask=True)
    val_dataset = CellCropsDataset(val_crops, transform=val_transform(crop_input_size), mask=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_channels = sum(1 for line in open(config["channels_path"])) + 1 - len(config["blacklist"])
    class_num = config["num_classes"]

    model = Model(num_channels + 1, class_num)

    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=16, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=16, shuffle=False)

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
        if i > 50 and patient >= 20:
            print('Loss has not decreased in last 5 epochs, Therefore, terminate training process.')
            break
