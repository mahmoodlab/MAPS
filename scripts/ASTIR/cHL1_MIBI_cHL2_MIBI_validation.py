import os
import argparse
import numpy as np
import pandas as pd
import yaml
from astir import Astir


def test(features, gt_labels, marker_dict, checkpoint_fold_dir, output_dir, split_name, class_names):
    ast = Astir(features, marker_dict)
    ast.load_model(os.path.join(checkpoint_fold_dir, "astir_summary.hdf5"))
    type_assignments = ast.predict_celltypes()
    save_results(type_assignments, gt_labels, split_name, output_dir, class_names)

def save_results(type_assignments, gt_labels, split_name, output_dir, class_names):
    type_assignments = type_assignments[class_names]
    pred_probs = type_assignments.to_numpy()
    pred_labels = np.argmax(pred_probs, axis=1)
    col_names = ["%s_prob" % col_name for col_name in type_assignments.columns.tolist()]
    df = pd.DataFrame(pred_probs, columns=col_names)
    df['pred_label'] = pred_labels
    df['gt_label'] = gt_labels
    df.to_csv(os.path.join(output_dir, "results_%s.csv" % split_name), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--data_dir', type=str, default='datasets/cHL1_MIBI/', help='configuration_path')
    parser.add_argument('--checkpoint_dir', type=str, default='results/cHL1_MIBI/', help='configuration_path')
    parser.add_argument('--result_dir', type=str, default='results/cHL1_MIBI_cHL2_MIBI/', help='configuration_path')
    args = parser.parse_args()

    for fid in range(5):
        fold_name = 'fold_%d' % fid
        train_cvs_path = os.path.join(args.data_dir, '%s' % fold_name, 'train.csv')
        valid_cvs_path = os.path.join(args.data_dir, '%s' % fold_name, 'valid.csv')
        test_cvs_path = os.path.join(args.data_dir, '%s' % fold_name, 'test.csv')
        marker_yaml_path = 'datasets/cHL1_MIBI/marker.yml'
        class_names = pd.read_csv('datasets/cHL1_MIBI/class_names.csv')['class_name'].tolist()
        checkpoint_fold_dir = os.path.join(args.checkpoint_dir, fold_name)
        output_dir = os.path.join(args.result_dir, fold_name)
        os.makedirs(output_dir, exist_ok=True)

        print(fold_name)
        with open(marker_yaml_path) as file:
            marker_dict = yaml.safe_load(file)
        cell_types = marker_dict['cell_types']

        train_df = pd.read_csv(train_cvs_path)
        train_labels = train_df['cell_label'].tolist()
        train_df = train_df.drop(columns=['cellSize', 'cell_label'])
        marker_names = train_df.columns.tolist()

        valid_df = pd.read_csv(valid_cvs_path)
        valid_labels = valid_df['cell_label'].tolist()
        valid_df = valid_df.drop(columns=['cellSize', 'cell_label'])

        test_df = pd.read_csv(test_cvs_path)
        test_labels = test_df['cell_label'].tolist()
        test_df = test_df.drop(columns=['cellSize', 'cell_label'])

        print('Marker Names: ', marker_names)
        print('Cell Type to Marker Mapping: ', cell_types)
        unique_marker_names = []
        for cell_type in cell_types.keys():
            for marker_name in cell_types[cell_type]:
                if marker_name not in unique_marker_names:
                    unique_marker_names.append(marker_name)
                if marker_name not in marker_names:
                    print(cell_type, marker_name)
        print(len(unique_marker_names), unique_marker_names)
        test(train_df[unique_marker_names], train_labels, marker_dict, checkpoint_fold_dir, output_dir, 'train', class_names)
        test(valid_df[unique_marker_names], valid_labels, marker_dict, checkpoint_fold_dir, output_dir, 'valid', class_names)
        test(test_df[unique_marker_names], test_labels, marker_dict, checkpoint_fold_dir, output_dir, 'test', class_names)
