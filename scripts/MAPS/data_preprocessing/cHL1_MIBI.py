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

    # data should be downloaded from the link given in the paper.
    data = pd.read_csv(os.path.join(args.data_csv_path))


    # Merge Cytotoxic CD8 cells to CD8 cells
    data.loc[data['Annotation'] == 'Cytotoxic CD8', 'Annotation'] = 'CD8'
    data.reset_index(inplace=True)


    # Extracting and Saving Marker Names from the data CSV File
    non_marker_cols = ['patientID', 'centroidX', 'centroidY', 'cellLabel', 'cellSize', 'Annotation', 'pointNum',
                       'tissue_block', 'region', 'index']
    marker_names = [col_name for col_name in data.columns.tolist() if col_name not in non_marker_cols]
    marker_names.sort()
    marker_df = pd.DataFrame(marker_names, columns=['marker_name'])
    marker_df.to_csv(os.path.join(args.output_dir, 'marker_names.csv'), index=False)


    # Creating Class ID and Name Mapping for Unique Classes in a Dataset

    class_dict = {'class_id': [], 'class_name': [], 'sample_count': []}
    class_mapping = {}
    for i, class_name in enumerate(np.unique(data['Annotation'].tolist())):
        class_dict['class_id'].append(i)
        class_dict['class_name'].append(class_name)
        class_dict['sample_count'].append(np.sum(data['Annotation'] == class_name))
        class_mapping[class_name] = i
    pd.DataFrame(class_dict).to_csv(os.path.join(args.output_dir, 'class_names.csv'), index=False)


    # Count number cells per patient per class
    class_names = np.unique(data['Annotation']).tolist()
    case_names = np.unique(data['patientID']).tolist()
    case_names = [case_name for case_name in case_names if 'ctrl' not in case_name]
    df = pd.DataFrame(np.zeros((len(case_names), len(class_names))), columns=class_names, index=case_names)
    for case_name, row in df.iterrows():
        for class_name in class_names:
            temp_df = data[data['patientID'] == case_name]
            temp_df = temp_df[temp_df['Annotation'] == class_name]
            df.loc[case_name, class_name] = temp_df.shape[0]
    df['Total'] = np.sum(df.to_numpy(), axis=1)
    df.to_csv(os.path.join(args.output_dir, 'class_count.csv'))


    # Assigning fold id to each patient based on number of cells
    df = df.sort_values(['Total'])
    case_to_fold_mapping = {'case_name': df.index.tolist(), 'fold_id': [-1] * df.shape[0]}

    i, j = 0, df.shape[0] - 1
    fid = 0
    fold_count = 5
    cell_per_class = df.to_numpy()
    cell_per_fold = np.zeros((fold_count, df.shape[1]))
    while i < j:
        case_to_fold_mapping['fold_id'][i] = fid % fold_count
        case_to_fold_mapping['fold_id'][j] = fid % fold_count
        cell_per_fold[fid % fold_count, :] += cell_per_class[i, :] + cell_per_class[j, :]
        fid += 1
        i += 1
        j -= 1
    cell_per_fold = pd.DataFrame(cell_per_fold, columns=df.columns)
    cell_per_fold.to_csv(os.path.join(args.output_dir, 'cell_count_per_fold.csv'))
    case_to_fold_mapping = pd.DataFrame(case_to_fold_mapping)
    case_to_fold_mapping.to_csv(os.path.join(args.output_dir, 'case_to_fold_mapping.csv'))

    # Generating training and validation splits using fold information
    features_cols = []
    features_cols.extend(marker_names)
    features_cols.append('cellSize')
    features_cols.append('patientID')
    features = data[features_cols]
    labels = np.array([class_mapping[class_name] for class_name in data['Annotation'].tolist()])
    features['cell_label'] = labels

    for i in range(fold_count):
        train_patient_list = case_to_fold_mapping[case_to_fold_mapping['fold_id'] != i]['case_name'].tolist()
        test_patient_list = case_to_fold_mapping[case_to_fold_mapping['fold_id'] == i]['case_name'].tolist()

        train_split = features[features['patientID'].isin(train_patient_list)]
        test_split = features[features['patientID'].isin(test_patient_list)]
        train_split = train_split.drop(columns=['patientID'])
        test_split = test_split.drop(columns=['patientID'])

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

    # Generating subset of training sets in each fold for training the model with different number of training samples
    for fid in range(5):
        train_csv = pd.read_csv(os.path.join(args.output_dir, 'splits', f"fold_{fid}", 'train.csv'))
        valid_csv = pd.read_csv(os.path.join(args.output_dir, 'splits', f"fold_{fid}", 'valid.csv'))
        test_csv = pd.read_csv(os.path.join(args.output_dir, 'splits', f"fold_{fid}", 'test.csv'))
        features = train_csv.to_numpy()[:, :-1]
        labels = train_csv.to_numpy()[:, -1]
        train_index = []
        skf = StratifiedKFold(n_splits=20, random_state=7325111, shuffle=True)
        for i, (_, train_index_) in enumerate(skf.split(features, labels)):
            train_index.extend(train_index_)
            if i in [0, 1, 4, 9, 14]:
                train = train_csv.iloc[train_index]
                print(
                    f"Fold {fid}: {(i + 1) * 5}%: {len(train_index) / features.shape[0]}, {train.shape[0] / features.shape[0]}")
                os.makedirs(os.path.join(args.output_dir, 'splits_%02d' % ((i + 1) * 5), f"fold_{fid}"), exist_ok=True)
                train.to_csv(os.path.join(args.output_dir, 'splits_%02d' % ((i + 1) * 5), f"fold_{fid}", 'train.csv'), index=False)
                valid_csv.to_csv(os.path.join(args.output_dir, 'splits_%02d' % ((i + 1) * 5), f"fold_{fid}", 'valid.csv'), index=False)
                test_csv.to_csv(os.path.join(args.output_dir, 'splits_%02d' % ((i + 1) * 5), f"fold_{fid}", 'test.csv'), index=False)


    # Generating train and valid sets with subset markers and classes
    selected_cols = ["CD11c", "CD163", "CD20", "CD4", "CD56", "CD68", "CD8", "CD3", "Pax-5", "cellSize", 'cell_label']
    
    pd.DataFrame(selected_cols, columns=['col_names']).to_csv(os.path.join(args.output_dir, 'selected_cols.csv'), index=False)
    selected_classes = ["CD4", "CD8", "DC", "M1", "M2", "NK", "Tumor", "B"]
    selected_labels = [class_mapping[class_name] for class_name in selected_classes]
    selected_labels_mapping = {class_mapping[class_name]: i for i, class_name in enumerate(selected_classes)}
    print(selected_labels_mapping)
    selected_labels_mapping[class_mapping['B']] = selected_classes.index('Tumor')
    print(selected_labels_mapping)
    class_dict = {'class_id': [], 'class_name': []}
    class_mapping = {}
    for i, class_name in enumerate(selected_classes):
        if class_name == 'B':
            continue
        class_dict['class_id'].append(i)
        class_dict['class_name'].append(class_name)
        class_mapping[class_name] = i
    pd.DataFrame(class_dict).to_csv(os.path.join(args.output_dir, 'selected_class_names.csv'), index=False)
    
    
    for fid in range(5):
        train = pd.read_csv(os.path.join(args.output_dir, 'splits', f"fold_{fid}", 'train.csv'))
        valid = pd.read_csv(os.path.join(args.output_dir, 'splits', f"fold_{fid}", 'valid.csv'))
    
        train = train[selected_cols]
        valid = valid[selected_cols]
    
        train = train[train['cell_label'].isin(selected_labels)]
        valid = valid[valid['cell_label'].isin(selected_labels)]
    
        train['cell_label'] = train['cell_label'].map(selected_labels_mapping)
        valid['cell_label'] = valid['cell_label'].map(selected_labels_mapping)
    
        os.makedirs(os.path.join(args.output_dir, 'splits_marker_subset', f"fold_{fid}"), exist_ok=True)
        train.to_csv(os.path.join(args.output_dir, 'splits_marker_subset', f"fold_{fid}", 'train.csv'), index=False)
        valid.to_csv(os.path.join(args.output_dir, 'splits_marker_subset', f"fold_{fid}", 'valid.csv'), index=False)