import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--data_csv_path', type=str,
                        help='path to the csv file containing the cell expression data of all cases along with their labels')
    parser.add_argument('--output_dir', type=str, help='directory to processed data files')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = pd.read_csv(args.data_csv_path)

    # Discard cells witl cell type 'Seg Artifact'
    dataset = dataset[dataset['cellType'] != 'Seg Artifact']

    # Merge Cytotoxic CD8 cells to CD8 cells
    dataset.loc[dataset['cellType'] == 'Cytotoxic CD8', 'cellType'] = 'CD8'

    dataset.reset_index(inplace=True, drop=True)

    # Creating Class ID and Name Mapping for Unique Classes in a Dataset

    class_dict = {'class_id': [], 'class_name': [], 'sample_count': []}
    class_mapping = {}
    for i, class_name in enumerate(np.unique(dataset['cellType'].tolist())):
        class_dict['class_id'].append(i)
        class_dict['class_name'].append(class_name)
        class_dict['sample_count'].append(np.sum(dataset['cellType'] == class_name))
        class_mapping[class_name] = i
    pd.DataFrame(class_dict).to_csv(os.path.join(args.output_dir, 'class_names.csv'), index=False)
    pd.DataFrame(class_dict)

    # Extracting and Saving Marker Names from a Dataset as a CSV File
    non_marker_cols = ['X_cent', 'Y_cent', 'cellLabel', 'cellSize', 'cellType']
    marker_names = [col_name for col_name in dataset.columns.tolist() if col_name not in non_marker_cols]
    marker_names.sort()
    pd.DataFrame(marker_names, columns=['marker_name']).to_csv(os.path.join(args.output_dir, 'marker_names.csv'), index=False)

    # Generating training and validation splits using fold information
    features_cols = []
    features_cols.extend(marker_names)
    features_cols.append('cellSize')
    features = dataset[features_cols]
    labels = np.array([class_mapping[class_name] for class_name in dataset['cellType'].tolist()])
    features['cell_label'] = labels

    skf = StratifiedKFold(n_splits=5, random_state=7325111, shuffle=True)
    for i, (train_index, valid_index) in enumerate(skf.split(features.to_numpy(), labels)):
        print(f"Fold {i}, {len(train_index)}, {len(valid_index)}:")

        train_split = features.iloc[train_index]
        test_split = features.iloc[valid_index]

        os.makedirs(os.path.join(args.output_dir, 'splits', 'fold_%d' % i), exist_ok=True)
        train_split.to_csv(os.path.join(args.output_dir, 'splits', 'fold_%d' % i, 'train.csv'), index=False)
        test_split.to_csv(os.path.join(args.output_dir, 'splits', 'fold_%d' % i, 'test.csv'), index=False)

    # Generating training, validation and test splits using fold information
    for fid in range(5):
        os.makedirs(os.path.join(args.output_dir, 'splits', f"fold_{fid}"), exist_ok=True)
        train_csv = pd.read_csv(os.path.join(args.output_dir, 'splits', f"fold_{fid}", 'train.csv'))
        features = train_csv.to_numpy()[:, :-1]
        labels = train_csv.to_numpy()[:, -1]
        skf = StratifiedKFold(n_splits=5, random_state=7325111, shuffle=True)
        for i, (train_index, valid_index) in enumerate(skf.split(features, labels)):
            train = train_csv.iloc[train_index]
            valid = train_csv.iloc[valid_index]
            print(f"Fold {fid}: {train.shape[0]}, {valid.shape[0]}")
            train.to_csv(os.path.join(args.output_dir, 'splits', f"fold_{fid}", 'train.csv'), index=False)
            valid.to_csv(os.path.join(args.output_dir, 'splits', f"fold_{fid}", 'valid.csv'), index=False)
            break
