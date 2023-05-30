import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SequentialSampler, WeightedRandomSampler


class CellExpressionCSV(Dataset):
    """
    A custom dataset class to read in and process cell expression data in CSV format.

    Args:
        csv_path (str): The path to the CSV file containing cell expression data.
        is_train (bool, optional): A flag indicating whether the data is used for training or not. Defaults to True.
        mean (numpy.ndarray, optional): The mean of the training data. Required if is_train is False. Defaults to None.
        std (numpy.ndarray, optional): The standard deviation of the training data. Required if is_train is False. Defaults to None.

    Attributes:
        x (numpy.ndarray): The cell expression data.
        y (numpy.ndarray): The cell labels, if present in the CSV file.
        is_train (bool): A flag indicating whether the data is used for training or not.
        mean (numpy.ndarray): The mean of the training data.
        std (numpy.ndarray): The standard deviation of the training data.
    """

    def __init__(self, csv_path, is_train=True, mean=None, std=None):
        self.x = None
        self.y = None
        self.is_train = is_train
        self.mean = mean
        self.std = std

        # Read the CSV file and extract cell expression data and labels
        df = pd.read_csv(csv_path)
        if 'cell_label' in df.columns.tolist():
            self.y = df['cell_label'].to_numpy()
            df = df.drop(columns=['cell_label'])
        self.x = df.to_numpy()

        # Normalize the cell expression data
        if is_train:
            if (self.mean is None) or (self.std is None):
                self.mean = np.mean(self.x, axis=0)
                self.std = np.std(self.x, axis=0)
                self.x = (self.x - self.mean) / self.std
            else:
                raise ValueError('Mean and/or std is None.')
        else:
            if (self.mean is not None) and (self.std is not None):
                self.x = (self.x - self.mean) / self.std

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Return the features and ground truth for a given index.

        Parameters:
        idx (int): Index of the data.

        Returns:
        feature (numpy.ndarray): Features of the data.
        gt (int): Ground truth label of the data. -1 if not available.
        """
        feature = self.x[idx] / 255.0
        if self.y is not None:
            gt = int(self.y[idx])
        else:
            gt = -1

        return feature, gt

    @staticmethod
    def get_data_loader(dataset, batch_size=4, is_train=False, num_workers=4):
        """
        Given a PyTorch dataset, this function returns a PyTorch dataloader.

        Parameters:
        - dataset (Dataset): PyTorch dataset to be loaded into a dataloader.
        - batch_size (int, optional): Number of samples in each batch. Defaults to 4.
        - is_train (bool, optional): Flag indicating whether the data is for training or not. Defaults to False.
        - num_workers (int, optional): Number of worker threads for data loading. Defaults to 4.

        Returns:
        - DataLoader: PyTorch dataloader containing the given dataset.
        """
        # If the data is for training, use WeightedRandomSampler to create a sampler for the data
        if is_train:
            labels = dataset.y.tolist()
            n = float(len(dataset))
            weight = [0] * int(n)
            weight_per_class = [n / labels.count(c) for c in np.unique(labels)]
            label_unique = np.unique(labels).tolist()
            for idx in range(len(dataset)):
                label = labels[idx]
                weight[idx] = weight_per_class[label_unique.index(label)]

            loader = DataLoader(dataset, batch_size=batch_size, sampler=WeightedRandomSampler(weight, len(weight)),
                                drop_last=True, num_workers=num_workers)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset), drop_last=False,
                                num_workers=num_workers)

        return loader
