import os
import argparse

from cell_phenotyping import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--dataset_dir', type=str,
                        help='dataset directory containing train and valid csv files for each fold')
    parser.add_argument('--results_dir', type=str, help='result directory to save model and logs')
    parser.add_argument('--fold_name', type=str, default='fold_0', help='fold name')
    parser.add_argument('--num_features', type=int, default='50', help='number of input features')
    parser.add_argument('--num_classes', type=int, default='16', help='number of classes')
    parser.add_argument('--batch_size', type=int, default='128', help='batch size')
    parser.add_argument('--learning_rate', type=float, default='0.001', help='learning rate')
    parser.add_argument('--dropout', type=float, default='0.10', help='dropout')
    parser.add_argument('--max_epochs', type=int, default='500', help='maximum number of epochs')
    parser.add_argument('--min_epochs', type=int, default='100', help='minimum number of epochs')
    parser.add_argument('--patience', type=int, default='50', help='patience')
    parser.add_argument('--seed', type=int, default='7325111', help='seed')
    parser.add_argument('--num_workers', type=int, default='4', help='number of workers')
    parser.add_argument('--verbose', type=int, default='1', help='verbose')

    args = parser.parse_args()

    print(args.fold_name)
    train_csv_path = os.path.join(args.dataset_dir, args.fold_name, 'train.csv')
    valid_csv_path = os.path.join(args.dataset_dir, args.fold_name, 'valid.csv')
    results_dir_fold = os.path.join(args.results_dir, args.fold_name)

    os.makedirs(results_dir_fold, exist_ok=True)

    model = Trainer(results_dir=results_dir_fold, num_features=args.num_features, num_classes=args.num_classes,
                    batch_size=args.batch_size, learning_rate=args.learning_rate, dropout=args.dropout,
                    max_epochs=args.max_epochs, min_epochs=args.min_epochs, patience=args.patience, seed=args.seed,
                    num_workers=args.num_workers, verbose=args.verbose)
    model.fit(train_csv_path, valid_csv_path)