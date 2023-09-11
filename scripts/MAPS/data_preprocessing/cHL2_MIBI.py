import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--data_csv_path', type=str,            
                        help='path to the directory containing csv files for each image')
    parser.add_argument('--output_dir', type=str,
                        help='directory to processed data files')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    image_list = ['2022-03-26T21-41-35_cHL_Slide302_S19_10582_5x5tile_2', '2022-03-27T02-42-50_cHL_Slide306_NIVO50A_5x5tile_1', '2022-03-30T01-38-11_cHL_Slide306_NIVO50A_5x5tile_4',
                  '2022-03-30T20-28-17_cHL_Slide308_NIVO51A_5x5tile_2', '2022-04-01T24-59-37_cHL_Slide307_NIVO50B_5x5tile_5', '2022-04-12T22-32-54_cHL_rLN-ctrl_Slide354_3x4tile']

    # Extracting and Saving Marker Names from the data CSV File
    non_marker_cols = ['cellLabel', 'X_cent', 'Y_cent', 'cellSize', 'pointNum', 'Annotation', 'Annotation_num', 'tile_name']
    data = pd.read_csv(os.path.join(args.data_csv_path, image_list[0] + '.csv'))
    marker_names = [col_name for col_name in data.columns.tolist() if col_name not in non_marker_cols]
    marker_names.sort()
    marker_df = pd.DataFrame(marker_names, columns=['marker_name'])
    marker_df.to_csv(os.path.join(args.output_dir, 'marker_names.csv'), index=False)

    # Case to Fold Mapping
    case_to_fold_mapping = {'Case Name': [], 'Fold ID': []}
    fid = 0
    for image_name in image_list:
        if 'ctrl' in image_name:
            continue
        else:
            case_to_fold_mapping['Case Name'].append(image_name)
            case_to_fold_mapping['Fold ID'].append(fid)
            fid += 1
    pd.DataFrame.from_dict(case_to_fold_mapping).to_csv(os.path.join(args.output_dir, 'case_to_fold_mapping.csv'))

    data = None
    for image_name in image_list:
        print(image_name)
        df = pd.read_csv(os.path.join(args.data_csv_path, '%s.csv' % image_name))
        df['image_name'] = image_name
        if data is None:
            data = df
        else:
            data = pd.concat([data, df], axis=0)

    # Creating Class ID and Name Mapping for Unique Classes in the Dataset
    class_dict = {'class_id': [], 'class_name': [], 'sample_count': []}
    class_mapping = {}
    for i, class_name in enumerate(np.unique(data['Annotation'].tolist())):
        class_dict['class_id'].append(i)
        class_dict['class_name'].append(class_name)
        class_dict['sample_count'].append(np.sum(data['Annotation'] == class_name))
        class_mapping[class_name] = i
    pd.DataFrame(class_dict).to_csv(os.path.join(args.output_dir, 'class_names.csv'), index=False)


    # Count number cells per image per class
    class_names = np.unique(data['Annotation']).tolist()
    case_names = np.unique(data['image_name']).tolist()
    # case_names = [case_name for case_name in case_names if 'ctrl' not in case_name]
    df = pd.DataFrame(np.zeros((len(case_names), len(class_names))), columns=class_names, index=case_names)
    for case_name, row in df.iterrows():
        for class_name in class_names:
            temp_df = data[data['image_name'] == case_name]
            temp_df = temp_df[temp_df['Annotation'] == class_name]
            df.loc[case_name, class_name] = temp_df.shape[0]
    df['Total'] = np.sum(df.to_numpy(), axis=1)
    df.to_csv(os.path.join(args.output_dir, 'class_count.csv'))

    # Generating training and validation splits using fold information
    features_cols = []
    features_cols.extend(marker_names)
    features_cols.append('cellSize')
    features_cols.append('image_name')
    features = data[features_cols]
    labels = np.array([class_mapping[class_name] for class_name in data['Annotation'].tolist()])
    features['cell_label'] = labels

    fid = 0
    for image_name in image_list:
        if 'ctrl' in image_name:
            continue
        else:
            train_split = features[features['image_name'] != image_name]
            test_split = features[features['image_name'] == image_name]
            train_split = train_split.drop(columns=['image_name'])
            test_split = test_split.drop(columns=['image_name'])

            print(fid, train_split.shape, test_split.shape)

            os.makedirs(os.path.join(args.output_dir, 'splits', 'fold_%d' % fid), exist_ok=True)
            train_split.to_csv(os.path.join(args.output_dir, 'splits', 'fold_%d' % fid, 'train.csv'), index=False)
            test_split.to_csv(os.path.join(args.output_dir, 'splits', 'fold_%d' % fid, 'test.csv'), index=False)
            fid += 1

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
