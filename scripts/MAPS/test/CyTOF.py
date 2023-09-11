import os
import argparse
import pandas as pd

from cell_phenotyping import Predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--data_csv_path', type=str, help='path to csv file containing data')
    parser.add_argument('--class_name_csv_path', type=str, help='path to csv file containing class names')
    parser.add_argument('--checkpoint_path', type=str, help='path to the checkpoint of the model for given fold')
    parser.add_argument('--results_dir', type=str, help='result directory to save model and logs')
    parser.add_argument('--num_features', type=int, default='32', help='number of input features')
    parser.add_argument('--num_classes', type=int, default='7', help='number of classes')
    parser.add_argument('--batch_size', type=int, default='128', help='batch size')
    parser.add_argument('--seed', type=int, default='7325111', help='seed')
    parser.add_argument('--num_workers', type=int, default='4', help='number of workers')
    parser.add_argument('--verbose', type=int, default='1', help='verbose')
    parser.add_argument('--normalization', type=bool, default='True', help='normalization')

    args = parser.parse_args()

    model = Predictor(model_checkpoint_path=args.checkpoint_path, num_features=args.num_features, num_classes=args.num_classes, batch_size=args.batch_size,
                      seed=args.seed, num_workers=args.num_workers, normalization=args.normalization, verbose=args.verbose)
    class_names = pd.read_csv(args.class_name_csv_path)['class_name'].tolist()
    col_names = [col_name + '_prob' for col_name in class_names]
    gt_labels = pd.read_csv(args.data_csv_path)['cell_label'].tolist()
    pred_labels, pred_probs = model.predict(args.data_csv_path)
    results = pd.DataFrame(pred_probs, columns=col_names)
    results['pred_label'] = pred_labels
    results['gt_label'] = gt_labels

    csv_name = os.path.basename(args.data_csv_path)
    os.makedirs(args.results_dir, exist_ok=True)
    results.to_csv(os.path.join(args.results_dir, 'results_%s' % csv_name), index=False)