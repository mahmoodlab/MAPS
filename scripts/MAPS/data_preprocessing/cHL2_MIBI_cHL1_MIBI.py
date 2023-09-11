import os
import argparse
import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--cHL1_class_name_csv_path', type=str, help='path to csv file containing class names for cHL1 dataset')
    parser.add_argument('--cHL2_class_name_csv_path', type=str, help='path to csv file containing class names for cHL2 dataset')
    parser.add_argument('--cHL1_marker_name_csv_path', type=str, help='path to csv file containing marker names for cHL1 dataset')
    parser.add_argument('--cHL2_marker_name_csv_path', type=str, help='path to csv file containing marker names for cHL2 dataset')
    parser.add_argument('--cHL2_data_csv_path', type=str, help='path to csv file containing data for cHL1 dataset')
    parser.add_argument('--output_dir', type=str, help='directory to processed data files')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cHL1_class_names = pd.read_csv(args.cHL1_class_name_csv_path)
    cHL2_class_names = pd.read_csv(args.cHL2_class_name_csv_path)

    cHL1_id_to_class_mapping = {}
    cHL1_class_to_id_mapping = {}
    for i, row in cHL1_class_names.iterrows():
        cHL1_id_to_class_mapping[row['class_id']] = row['class_name']
        cHL1_class_to_id_mapping[row['class_name']] = row['class_id']

    cHL2_id_to_class_mapping = {}
    cHL2_class_to_id_mapping = {}
    for i, row in cHL1_class_names.iterrows():
        cHL2_id_to_class_mapping[row['class_id']] = row['class_name']
        cHL2_class_to_id_mapping[row['class_name']] = row['class_id']

    cHL2_to_cHL1_class_name_mapping = {'B': 'B', 
                                       'CD4 T': 'CD4', 
                                       'CD8 T': 'CD8', 
                                       'DC': 'DC',
                                       'Endothelial': 'Endothelial', 
                                       'M1': 'M1', 
                                       'M2': 'M2', 
                                       'NK': 'NK',
                                       'Neutrophil': 'Neutrophil', 
                                       'Other': 'Other', 
                                       'CD4 Treg': 'Treg',
                                       'Tumor': 'Tumor'}
    cHL1_to_cHL2_class_name_mapping = {'B': 'B', 
                                       'CD4': 'CD4 T', 
                                       'CD8': 'CD8 T', 
                                       'Cytotoxic CD4': 'CD4 T', 
                                       'DC': 'DC',
                                       'Endothelial': 'Endothelial', 
                                       'M1': 'M1', 
                                       'M2': 'M2', 
                                       'NK': 'NK',
                                       'Neutrophil': 'Neutrophil', 
                                       'Other': 'Other', 
                                       'Treg': 'CD4 Treg',
                                       'Tumor': 'Tumor'}

    cHL2_to_cHL1_class_id_mapping = {}
    for key in cHL2_to_cHL1_class_name_mapping.keys():
        cHL2_class_id = cHL2_to_cHL1_class_name_mapping[key]
        cHL1_class_id = cHL1_class_to_id_mapping[cHL2_to_cHL1_class_name_mapping[key]]
        cHL2_to_cHL1_class_name_mapping[cHL2_class_id] = cHL1_class_id

    cHL2_to_cHL1_class_id_mapping = {}
    for key in cHL2_to_cHL1_class_name_mapping.keys():
        cHL2_class_id = cHL2_class_to_id_mapping[key]
        cHL1_class_id = cHL1_class_to_id_mapping[cHL2_to_cHL1_class_name_mapping[key]]
        cHL2_to_cHL1_class_id_mapping[cHL2_class_id] = cHL1_class_id

    cHL1_marker_names = pd.read_csv(args.cHL1_marker_name_csv_path)
    cHL2_marker_names = pd.read_csv(args.cHL2_marker_name_csv_path)

    cHL2_data = pd.read_csv(args.cHL2_data_csv_path)
    cHL2_data['cell_label'] = cHL2_data['cell_label'].apply(lambda x: cHL2_to_cHL1_class_id_mapping[x])

    for marker_name in cHL1_marker_names:
        if marker_name not in cHL2_data.columns:
            cHL2_data[marker_name] = 0.0
    if 'cellSize' not in nolan_marker_names:
        nolan_marker_names.append('cellSize')
    df = df[nolan_marker_names]

    csv_name = os.path.basename(args.cHL2_data_csv_path)
    cHL2_data.to_csv(os.path.join(args.output_dir, csv_name), index=False)
