import os
import argparse
import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--data_csv_path', type=str,
                        help='path to the csv file containing the cell expression data of all cases along with their labels')
    parser.add_argument('--output_dir', type=str, help='directory to processed data files')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    class_names = ['B cells', 'CD11c+ DCs', 'CD4+ T cells', 'CD8+ T cells', 'Granulocytes', 'Immune cells', 'Macrophages', 'NK cells', 'Plasma cells', 'Smooth muscle', 'Stroma', 'Tregs', 'Tumor cells', 'Vasculature']
    class_mapping = {'B cells': 'B cells',
                     'CD11b+CD68+ macrophages': 'Macrophages',
                     'CD11c+ DCs': 'CD11c+ DCs',
                     'CD163+ macrophages': 'Macrophages',
                     'CD4+ T cells': 'CD4+ T cells',
                     'CD4+ T cells CD45RO+': 'CD4+ T cells',
                     'CD4+ T cells GATA3+': 'CD4+ T cells',
                     'CD68+ macrophages': 'Macrophages',
                     'CD68+ macrophages GzmB+': 'Macrophages',
                     'CD68+CD163+ macrophages': 'Macrophages',
                     'CD8+ T cells': 'CD8+ T cells',
                     'NK cells': 'NK cells',
                     'Tregs': 'Tregs',
                     'granulocytes': 'Granulocytes',
                     'immune cells': 'Immune cells',
                     'plasma cells': 'Plasma cells',
                     'smooth muscle': 'Smooth muscle',
                     'stroma': 'Stroma',
                     'tumor cells': 'Tumor cells',
                     'vasculature': 'Vasculature'}
    marker_names = [
        "aSMA - smooth muscle:Cyc_11_ch_2",
        "beta-catenin - Wnt signaling:Cyc_4_ch_4",
        "CD11b - macrophages:Cyc_10_ch_3",
        "CD11c - DCs:Cyc_12_ch_3",
        "CD138 - plasma cells:Cyc_21_ch_3",
        "CD15 - granulocytes:Cyc_14_ch_2",
        "CD163 - macrophages:Cyc_17_ch_3",
        "CD20 - B cells:Cyc_8_ch_3",
        "CD21 - DCs:Cyc_6_ch_4",
        "CD3 - T cells:Cyc_16_ch_4",
        "CD31 - vasculature:Cyc_19_ch_3",
        "CD34 - vasculature:Cyc_20_ch_3",
        "CD38 - multifunctional:Cyc_20_ch_4",
        "CD4 - T helper cells:Cyc_6_ch_3",
        "CD44 - stroma:Cyc_2_ch_2",
        "CD45 - hematopoietic cells:Cyc_4_ch_2",
        "CD45RA - naive T cells:Cyc_6_ch_2",
        "CD45RO - memory cells:Cyc_18_ch_3",
        "CD56 - NK cells:Cyc_10_ch_4",
        "CD68 - macrophages:Cyc_18_ch_4",
        "CD7 - T cells:Cyc_16_ch_3",
        "CD8 - cytotoxic T cells:Cyc_3_ch_2",
        "Chromogranin A - neuroendocrine:Cyc_17_ch_2",
        "Cytokeratin - epithelia:Cyc_10_ch_2",
        "FOXP3 - regulatory T cells:Cyc_2_ch_3",
        "GATA3 - Th2 helper T cells:Cyc_3_ch_4",
        "GFAP - nerves:Cyc_16_ch_2",
        "Granzyme B - cytotoxicity:Cyc_13_ch_2",
        "HLA-DR - MHC-II:Cyc_5_ch_2",
        "MUC-1 - epithelia:Cyc_7_ch_2",
        "Na-K-ATPase - membranes:Cyc_9_ch_2",
        "p53 - tumor suppressor:Cyc_3_ch_3",
        "Podoplanin - lymphatics:Cyc_19_ch_4",
        "Synaptophysin - neuroendocrine:Cyc_15_ch_3"
    ]
    pd.DataFrame(marker_names, columns=['marker_name']).to_csv(os.path.join(args.output_dir, 'marker_names.csv'), index=False)

    # Adding Class ID Column to Dataset
    dataset = pd.read_csv(args.data_csv_path)
    dataset = dataset[dataset['ClusterName'].isin(class_mapping.keys())]
    dataset.reset_index(inplace=True)
    dataset['cell_label'] = -1
    for i, row in dataset.iterrows():
        class_name = class_mapping[row['ClusterName']]
        dataset.loc[i, 'cell_label'] = class_names.index(class_name)

    # Cell Count per Class
    class_count = {'class_id': [], 'class_name': [], 'class_count': []}
    for i in range(len(class_names)):
        class_count['class_id'].append(i)
        class_count['class_name'].append(class_names[i])
        class_count['class_count'].append(dataset[dataset['cell_label'] == i].shape[0])
    df = pd.DataFrame.from_dict(class_count)
    df.to_csv(os.path.join(args.output_dir, 'class_names.csv'), index=False)

    # Cell Count per Spot per Class
    cell_per_spot = {'Spot_Name': []}
    for i in range(len(class_names)):
        cell_per_spot[class_names[i]] = []
    cell_per_spot['Total'] = []
    for spot_name in np.unique(dataset['spots']):
        cell_per_spot['Spot_Name'].append(spot_name)
        df = dataset[dataset['spots'] == spot_name]
        cell_per_spot['Total'].append(df.shape[0])
        for i in range(len(class_names)):
            cell_count = df[df['cell_label'] == i].shape[0]
            cell_per_spot[class_names[i]].append(cell_count)
    df = pd.DataFrame.from_dict(cell_per_spot)
    df.sort_values('Total', inplace=True)
    df.to_csv(os.path.join(args.output_dir, 'cells_per_spot.csv'), index=False)

    # Spot to Fold Mapping
    spot_to_fold_mapping = {'spot_name': df.index.tolist(), 'fold_id': [-1] * df.shape[0]}

    i, j = 0, df.shape[0] - 1
    fid = 0
    fold_count = 5
    cell_per_class = df.to_numpy()
    cell_per_fold = np.zeros((fold_count, df.shape[1]))
    while i < j:
        spot_to_fold_mapping['fold_id'][i] = fid % fold_count
        spot_to_fold_mapping['fold_id'][j] = fid % fold_count
        cell_per_fold[fid % fold_count, :] += cell_per_class[i, :] + cell_per_class[j, :]
        fid += 1
        i += 1
        j -= 1
    cell_per_fold = pd.DataFrame(cell_per_fold, columns=df.columns)
    cell_per_fold.to_csv(os.path.join(args.output_dir, 'cell_count_per_fold.csv'))
    spot_to_fold_mapping = pd.DataFrame(spot_to_fold_mapping)
    spot_to_fold_mapping.to_csv(os.path.join(args.output_dir, 'spot_to_fold_mapping.csv'))

    # Generating training and validation splits using fold information
    features_cols = []
    features_cols.extend(marker_names)
    features_cols.append('spots')
    features_cols.append('cell_label')
    features = dataset[features_cols]
    labels = features['cell_label'].tolist()

    for i in range(fold_count):
        train_patient_list = spot_to_fold_mapping[spot_to_fold_mapping['fold_id'] != i]['spot_name'].tolist()
        valid_patient_list = spot_to_fold_mapping[spot_to_fold_mapping['fold_id'] == i]['spot_name'].tolist()

        train_split = features[features['spots'].isin(train_patient_list)]
        valid_split = features[features['spots'].isin(valid_patient_list)]
        train_split = train_split.drop(columns=['spots'])
        valid_split = valid_split.drop(columns=['spots'])

        print(i, train_split.shape, valid_split.shape)

        os.makedirs(os.path.join(args.output_dir, 'splits', 'fold_%d' % i), exist_ok=True)
        train_split.to_csv(os.path.join(args.output_dir, 'splits', 'fold_%d' % i, 'train.csv'), index=False)
        valid_split.to_csv(os.path.join(args.output_dir, 'splits', 'fold_%d' % i, 'valid.csv'), index=False)
