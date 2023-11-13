from __future__ import print_function
import os
import sys
import random
import numpy as np

import torch

from .datasets import CellExpressionCSV
from .networks import MLP
from .trainer import Trainer


class Predictor(Trainer):
    """
    Class to predict a cell phenotype using pretrained classification model.

    Attributes:
    ----------
    model_checkpoint_path: str, optional
        Path to a pre-trained model checkpoint. Default is None.
    results_dir: str, optional
        Path to directory to store the training results. Default is './results/'.
    num_features: int, optional
        The number of features in the input data. Default is 47.
    num_classes: int, optional
        The number of classes in the classification task. Default is 12.
    batch_size: int, optional
        The batch size used for training and evaluation. Default is 128.
    seed: int, optional
        The seed used to initialize the random number generator. Default is 7325111.
    num_workers: int, optional
        The number of workers used in the data loading. Default is 4.
    normalization : bool, optional
        Flag to indicate whether to normalize data before inference using pre-calculated mean and standard deviation 
        from training dataset. Default is True.
    verbose: int, optional
        The verbosity level. Default is 0.

    """

    def __init__(self, model_checkpoint_path=None, results_dir='./results/', num_features=47, num_classes=12,
                 batch_size=128, seed=7325111, num_workers=4, normalization=True, verbose=0):
        """
        Initialize the CellPhenotypeTrainer class.
        """
        super().__init__(model_checkpoint_path=model_checkpoint_path,
                         results_dir=results_dir,
                         num_features=num_features,
                         num_classes=num_classes,
                         batch_size=batch_size,
                         seed=seed,
                         num_workers=num_workers,
                         verbose=verbose)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed()
        self.init_model()
        self.load_model()
        if normalization:
            self.train_data_mean, self.train_data_std = self.load_model()
        else:
            self.train_data_mean, self.train_data_std = None, None

    def predict(self, test_data_path):
        """
        This function is used to make predictions on the test data.

        Parameters:
        test_data_path (str): path to the test data

        Returns:
        pred_labels (numpy.ndarray): List of predicted class labels
        pred_probs (numpy.ndarray): List of predicted class probabilities
        """

        data_loader = self.init_data_loader(test_data_path, mean=self.train_data_mean, std=self.train_data_std)

        pred_labels, pred_probs = self.test_loop(data_loader)

        return pred_labels, pred_probs
