import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--data_csv_path', type=str,
                        help='path to the csv file containing the cell expression data of all cases along with their labels')
    parser.add_argument('--output_dir', type=str,help='directory to processed data files')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # data should be downloaded from the link given in the paper.
    data = pd.read_csv(os.path.join(args.data_csv_path))
    data = data.dropna()
    data = data.reset_index(drop=True)

    # Extracting and Saving Marker Names from the data CSV File
    non_marker_cols = ['cell_index', 'x', 'y', 'sample_name', 'frame_name', 'phenotype_label', 'cellSize']
    marker_names = [col_name for col_name in data.columns.tolist() if col_name not in non_marker_cols]
    marker_names.sort()
    marker_df = pd.DataFrame(marker_names, columns=['marker_name'])
    marker_df.to_csv(os.path.join(args.output_dir, 'marker_names.csv'), index=False)

    # Creating Class ID and Name Mapping for Unique Classes in a Dataset

    class_dict = {'class_id': [], 'class_name': [], 'sample_count': []}
    class_mapping = {}
    for i, class_name in enumerate(np.unique(data['phenotype_label'].tolist())):
        class_dict['class_id'].append(i)
        class_dict['class_name'].append(class_name)
        class_dict['sample_count'].append(np.sum(data['phenotype_label'] == class_name))
        class_mapping[class_name] = i
    pd.DataFrame(class_dict).to_csv(os.path.join(args.output_dir, 'class_names.csv'), index=False)

    # Count number cells per patient per class
    class_names = np.unique(data['phenotype_label']).tolist()
    sample_names = np.unique(data['sample_name']).tolist()
    sample_names = [sample_name for sample_name in sample_names if 'rLN' not in sample_name]
    df = pd.DataFrame(np.zeros((len(sample_names), len(class_names))), columns=class_names, index=sample_names)
    for sample_name, row in df.iterrows():
        for class_name in class_names:
            temp_df = data[data['sample_name'] == sample_name]
            temp_df = temp_df[temp_df['phenotype_label'] == class_name]
            df.loc[sample_name, class_name] = temp_df.shape[0]
    df['Total'] = np.sum(df.to_numpy(), axis=1)
    df.to_csv(os.path.join(args.output_dir, 'class_count.csv'))

    # Assigning fold id to each patient based on number of cells
    df = df.sort_values(['Total'])
    sample_to_fold_mapping = {'sample_name': df.index.tolist(), 'fold_id': [-1] * df.shape[0]}

    i, j = 0, df.shape[0] - 1
    fid = 0
    fold_count = 5
    cell_per_class = df.to_numpy()
    cell_per_fold = np.zeros((fold_count, df.shape[1]))
    while i < j:
        sample_to_fold_mapping['fold_id'][i] = fid % fold_count
        sample_to_fold_mapping['fold_id'][j] = fid % fold_count
        cell_per_fold[fid % fold_count, :] += cell_per_class[i, :] + cell_per_class[j, :]
        fid += 1
        i += 1
        j -= 1
    cell_per_fold = pd.DataFrame(cell_per_fold, columns=df.columns)
    cell_per_fold.to_csv(os.path.join(args.output_dir, 'cell_count_per_fold.csv'))
    sample_to_fold_mapping = pd.DataFrame(sample_to_fold_mapping)
    sample_to_fold_mapping.to_csv(os.path.join(args.output_dir, 'sample_to_fold_mapping.csv'))

    sample_to_fold_mapping_df = pd.read_csv(os.path.join(args.output_dir, 'sample_to_fold_mapping.csv'))
    fov_to_fold_mapping = {'frame_name': [], 'fold_id': []}
    ctrl_fov = {'frame_name': []}
    for i, row in sample_to_fold_mapping_df.iterrows():
        sample_name = row['sample_name']
        fold_id = row['fold_id']
        temp_df = data[data['sample_name'] == sample_name]
        fov_to_fold_mapping['frame_name'].extend(temp_df['frame_name'].tolist())
        fov_to_fold_mapping['fold_id'].extend([fold_id] * temp_df.shape[0])

    temp_df = data[data['sample_name'].isin(sample_to_fold_mapping_df['sample_name'].tolist()) == False]
    ctrl_fov['frame_name'].extend(temp_df['frame_name'].tolist())

    temp_df = pd.DataFrame.from_dict(fov_to_fold_mapping)
    temp_df.drop_duplicates(inplace=True)
    temp_df.to_csv(os.path.join(args.output_dir, 'fov_to_fold_mapping.csv'), index=False)
    temp_df = pd.DataFrame.from_dict(ctrl_fov)
    temp_df.drop_duplicates(inplace=True)
    temp_df.to_csv(os.path.join(args.output_dir, 'ctrl_fov.csv'), index=False)

    # Generating training and validation splits using fold information
    features_cols = []
    features_cols.extend(marker_names)
    features_cols.append('cellSize')
    features_cols.append('sample_name')
    features = data[features_cols]
    labels = np.array([class_mapping[class_name] for class_name in data['phenotype_label'].tolist()])
    features['cell_label'] = labels

    for i in range(fold_count):
        train_patient_list = sample_to_fold_mapping[sample_to_fold_mapping['fold_id'] != i]['sample_name'].tolist()
        test_patient_list = sample_to_fold_mapping[sample_to_fold_mapping['fold_id'] == i]['sample_name'].tolist()

        train_split = features[features['sample_name'].isin(train_patient_list)]
        test_split = features[features['sample_name'].isin(test_patient_list)]
        train_split = train_split.drop(columns=['sample_name'])
        test_split = test_split.drop(columns=['sample_name'])

        print(i, train_split.shape, test_split.shape)

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