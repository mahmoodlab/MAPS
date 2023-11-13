#!/bin/bash

root_dir='~/MAPS/cHL2_MIBI/'
data_csv_path='path_to_data_dir/cHL_2_MIBI/annotation_csv/'
metadata_csv_path='path_to_data_dir/cHL_2_MIBI/cHL2_metadata.csv'
dataset_dir="$root_dir/datasets"
results_dir="$root_dir/results"
num_features=47
num_classes=12
batch_size=128
learning_rate=0.001
dropout=0.25
max_epochs=500
min_epochs=100
patience=50
seed=7325111
num_workers=4
verbose=1
fold_names=("fold_0" "fold_1" "fold_2" "fold_3" "fold_4")

# Data preprocessing
python data_preprocessing/cHL2_MIBI.py --data_csv_path "$data_csv_path" --metadata_csv_path "$metadata_csv_path" --output_dir "$dataset_dir"

# Model training

for fold_name in "${fold_names[@]}"
do
    python train/cHL2_MIBI.py \
    --dataset_dir "$dataset_dir/splits" \
    --results_dir "$results_dir/train_valid_test/" \
    --fold_name "$fold_name" \
    --num_features "$num_features" \
    --num_classes "$num_classes" \
    --batch_size "$batch_size" \
    --learning_rate "$learning_rate" \
    --dropout "$dropout" \
    --max_epochs "$max_epochs" \
    --min_epochs "$min_epochs" \
    --patience "$patience" \
    --seed "$seed" \
    --num_workers "$num_workers" \
    --verbose "$verbose"
done

for fold_name in "${fold_names[@]}"
do
    python test/cHL2_MIBI.py \
    --data_csv_path "$dataset_dir/splits/$fold_name/test.csv" \
    --class_name_csv_path "$dataset_dir/class_names.csv" \
    --checkpoint_path "$results_dir/train_valid_test/$fold_name/best_checkpoint.pt" \
    --results_dir "$results_dir/train_valid_test/$fold_name" \
    --num_features "$num_features" \
    --num_classes "$num_classes" \
    --batch_size "$batch_size" \
    --seed "$seed" \
    --num_workers "$num_workers" \
    --verbose "$verbose"
done