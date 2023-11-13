import pandas as pd

val_results = [] #Fill here the paths to all of yours val_results.csv files you got from the training/validation

df_all_labeled = pd.DataFrame()
ensemble_size = len(val_results)
for i, val_result in enumerate(val_results):
    curr_df = pd.read_csv(val_result, index_col=0)
    prob_list = curr_df["prob_list"].apply(eval)
    num_classes = len(prob_list.iloc[0])
    curr_df[[f"prob_class_{j}" for j in range(num_classes)]] = prob_list.apply(pd.Series)
    curr_df.columns = [c+f"_ens_{i}" for c in curr_df.columns]
    df_all_labeled = pd.concat([df_all_labeled, curr_df], axis=1)

for i in range(num_classes):
    df_all_labeled[f"prob_mean_class_{i}"] = df_all_labeled[[f"prob_class_{i}_ens_{j}" for j in range(ensemble_size)]].mean(axis=1)

df_all_labeled["pred"] = df_all_labeled[[f"prob_mean_class_{i}" for i in range(num_classes)]].values.argmax(1)
df_all_labeled["pred_prob"] = df_all_labeled[[f"prob_mean_class_{i}" for i in range(num_classes)]].max(axis=1)

df_all_labeled["label"] = df_all_labeled["label_ens_1"]
df_all_labeled["cell_id"] = df_all_labeled["cell_id_ens_1"]
df_all_labeled["image_id"] = df_all_labeled["image_id_ens_1"]

df_all_labeled[["image_id", "cell_id", "label", "pred", "pred_prob"]].to_csv("merged_ensemble.csv")