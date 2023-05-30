from __future__ import print_function
import os
import sys
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from .datasets import CellExpressionCSV
from .networks import MLP


class Trainer:
    """
    Class to train a cell phenotype classification model.

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
    learning_rate: float, optional
        The learning rate for the Adam optimizer. Default is 0.001.
    dropout: float, optional
        The dropout rate used in the MLP network. Default is 0.10.
    max_epochs: int, optional
        The maximum number of epochs to run the training. Default is 500.
    min_epochs: int, optional
        The minimum number of epochs to run the training. Default is 250.
    patience: int, optional
        The patience used in early stopping. Default is 100.
    seed: int, optional
        The seed used to initialize the random number generator. Default is 7325111.
    num_workers: int, optional
        The number of workers used in the data loading. Default is 4.
    verbose: int, optional
        The verbosity level. Default is 0.

    """

    def __init__(self, model_checkpoint_path=None, results_dir='./results/', num_features=47, num_classes=12,
                 batch_size=128, learning_rate=0.001, dropout=0.10, max_epochs=500,
                 min_epochs=250, patience=100, seed=7325111, num_workers=4, verbose=0):
        """
        Initialize the CellPhenotypeTrainer class.
        """
        self.model_checkpoint_path = model_checkpoint_path
        self.results_dir = results_dir
        self.num_features = num_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.patience = patience
        self.num_workers = num_workers
        self.seed = seed
        self.verbose = verbose

        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.counter = 0
        self.lowest_loss = np.Inf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_seed(self):
        """
        Sets the seed for the random number generator.

        This function sets the seed for both the NumPy and PyTorch random number generators, ensuring that results are reproducible.

        Parameters:
        seed (int): The seed to be set for the random number generators.

        Returns:
        None
        """
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if str(self.device.type) == 'cuda':
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def init_data_loader(self, data_path, is_train=False, mean=None, std=None):
        """
        Initialize and return the data loader for the given data path.

        Parameters:
        - data_path (str): Path to the data file.
        - is_train (bool, optional): Whether to return the data loader for training or not. Default is False.
        - mean (list, optional): List of mean values to be used for normalization. Default is None.
        - std (list, optional): List of standard deviation values to be used for normalization. Default is None.

        Returns:
        - torch.utils.data.DataLoader: The data loader for the given data path.
        """
        dataset = CellExpressionCSV(data_path, is_train=is_train, mean=mean, std=std)
        return CellExpressionCSV.get_data_loader(dataset, batch_size=self.batch_size, is_train=is_train,
                                                 num_workers=self.num_workers)

    def init_model(self):
        """
        Initialize the Multi-layer Perceptron (MLP) model for cell expression classification.

        Parameters:
        None

        Returns:
        None

        """
        self.model = MLP(input_dim=self.num_features, hidden_dim=512, num_classes=self.num_classes,
                         dropout=self.dropout)
        self.model.to(self.device, dtype=torch.float64)

    def init_optimizer(self):
        """
        Initialize the optimizer for the model.

        Returns:
            None
        """
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)

    def init_loss_function(self):
        """
        Initialize the loss function for the model.

        Returns:
            None
        """
        self.loss_fn = nn.CrossEntropyLoss()

    def save_model(self, mean, std):
        """
        This function saves the current state of the model along with mean and standard deviation of the train data.

        Parameters:
        mean (float): mean of the train data
        std (float): standard deviation of the train data

        Returns:
        None
        """
        os.makedirs(self.results_dir, exist_ok=True)  # Creating results directory if it is not there.
        save_dict = {'model_parameters': self.model.state_dict(), 'train_data_mean': mean, 'train_data_std': std}
        torch.save(save_dict, os.path.join(self.results_dir, 'best_checkpoint.pt'))
        self.model_checkpoint_path = os.path.join(self.results_dir, 'best_checkpoint.pt')

    def load_model(self):
        """
        This function loads the saved state of the model along with mean and standard deviation of the train data.

        Returns:
        mean (float): mean of the train data
        std (float): standard deviation of the train data
        """
        if self.model_checkpoint_path is None:
            raise ValueError('Model Checkpoint path is None.')
        else:
            ckpt_path = self.model_checkpoint_path
            loaded_dict = torch.load(ckpt_path)
            self.model.load_state_dict(loaded_dict['model_parameters'], strict=True)
        return loaded_dict['train_data_mean'], loaded_dict['train_data_std']

    def fit(self, train_dataset_path, valid_dataset_path):
        """
        Trains the model with the provided training dataset and validation dataset
        and saves the best model based on the lowest validation loss.
        The training progress, including loss, accuracy, and AUC, is recorded
        and saved in a CSV file in the "results" directory.

        Parameters:
        train_dataset_path (str): Path to the training dataset.
        valid_dataset_path (str): Path to the validation dataset.

        Returns:
        None
        """

        self.counter = 0
        self.lowest_loss = np.Inf
        self.set_seed()
        self.init_model()
        self.init_optimizer()
        self.init_loss_function()
        train_dl = self.init_data_loader(train_dataset_path, is_train=True)
        train_dataset_mean = train_dl.dataset.mean
        train_data_std = train_dl.dataset.std
        valid_dl = self.init_data_loader(valid_dataset_path, mean=train_dataset_mean, std=train_data_std)

        result_dict = {'train_loss': [], 'valid_loss': [],
                       'train_acc': [], 'valid_acc': [],
                       'train_auc': [], 'valid_auc': []}
        for epoch in range(self.max_epochs):
            start_time = time.time()

            # Train the model for one epoch
            train_loss, train_acc, train_auc = self.train_loop(train_dl)
            print('\rTrain Epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}, train_auc: {:.4f}                 '.format(
                epoch, train_loss, train_acc, train_auc))
            # Record the training loss, accuracy, and AUC
            result_dict['train_loss'].append(train_loss)
            result_dict['train_acc'].append(train_acc)
            result_dict['train_auc'].append(train_auc)

            # Evaluate the model on the validation dataset
            valid_loss, valid_acc, valid_auc = self.valid_loop(valid_dl)
            print('\rValid Epoch: {}, valid_loss: {:.4f}, valid_acc: {:.4f}, valid_auc: {:.4f}                 '.format(
                epoch, valid_loss, valid_acc, valid_auc))
            # Record the validation loss, accuracy, and AUC
            result_dict['valid_loss'].append(valid_loss)
            result_dict['valid_acc'].append(valid_acc)
            result_dict['valid_auc'].append(valid_auc)

            # Save the model with the lowest validation loss
            if self.lowest_loss > valid_loss:
                print('--------------------Saving best model--------------------')
                self.save_model(train_dataset_mean, train_data_std)
                self.lowest_loss = valid_loss
                self.counter = 0
            else:
                self.counter += 1
                print('Loss is not decreased in last %d epochs' % self.counter)

            if (self.counter > self.patience) and (epoch >= self.min_epochs):
                break

            total_time = time.time() - start_time
            print('Time to process epoch({}): {:.4f} minutes                             \n'.format(epoch, total_time/60))

            pd.DataFrame.from_dict(result_dict).to_csv(os.path.join(self.results_dir, 'training_logs.csv'))

    def predict(self, test_data_path):
        """
        This function is used to make predictions on the test data.

        Parameters:
        test_data_path (str): path to the test data

        Returns:
        pred_labels (numpy.ndarray): List of predicted class labels
        pred_probs (numpy.ndarray): List of predicted class probabilities
        """
        self.set_seed()
        self.init_model()
        self.load_model()
        train_data_mean, train_data_std = self.load_model()

        data_loader = self.init_data_loader(test_data_path, mean=train_data_mean, std=train_data_std)

        pred_labels, pred_probs = self.test_loop(data_loader)

        return pred_labels, pred_probs

    def train_loop(self, data_loader):
        """
        Train the model using the training data provided in the `data_loader` argument.

        Parameters:
        data_loader (torch.utils.data.DataLoader): The PyTorch data loader containing the training data.

        Returns:
        - avg_loss: The average training loss for the epoch.
        - acc: The accuracy of the model on the training data.
        - auc: The area under the receiver operating characteristic curve (AUC) of the model on the training data.
        """
        total_loss = 0
        gt_labels = []
        pred_labels = []
        pred_probs = None
        batch_count = len(data_loader)
        self.model.train()
        for batch_idx, (features_batch, label_batch) in enumerate(data_loader):
            # Convert ground-truth labels to a list
            if len(gt_labels) == 0:
                gt_labels = label_batch.cpu().numpy().tolist()
            else:
                gt_labels.extend(label_batch.cpu().numpy().tolist())

            # Transfer features and labels to the GPU if one is available
            features_batch = features_batch.to(self.device)
            label_batch = label_batch.to(self.device)

            # Forward pass through the model
            logits, probs = self.model(features_batch)

            # Calculate the loss and backpropagate the gradients
            self.optimizer.zero_grad()
            loss = self.loss_fn(logits, label_batch)
            loss.backward()
            self.optimizer.step()

            # Accumulate the loss
            total_loss += loss.cpu().item()

            # Convert predicted labels and probabilities to lists
            p_labels = torch.argmax(probs, dim=1)
            p_labels = p_labels.detach().cpu().numpy().tolist()
            p_probs = probs.detach().cpu().numpy()

            # Accumulate the predicted labels and probabilities
            if len(pred_labels) == 0:
                pred_probs = p_probs
                pred_labels = p_labels
            else:
                pred_probs = np.concatenate((pred_probs, p_probs), axis=0)
                pred_labels.extend(p_labels)

            # Output the training progress
            sys.stdout.write('\rTraining Batch {}/{}, avg loss: {:.4f}'.format(
                batch_idx + 1, batch_count, total_loss / (batch_idx + 1)))

        # Compute accuracy and AUC of the model
        acc = accuracy_score(gt_labels, pred_labels)
        auc = roc_auc_score(gt_labels, pred_probs, multi_class='ovo', labels=[i for i in range(self.num_classes)])

        # Output the confusion matrix if verbosity is set to 1
        if self.verbose == 1:
            print('\nConfusion Matrix')
            conf = pd.DataFrame(confusion_matrix(gt_labels, pred_labels, labels=[i for i in range(self.num_classes)]),
                                index=['class_%d' % i for i in range(self.num_classes)],
                                columns=['class_%d' % i for i in range(self.num_classes)])
            print(conf)
        return total_loss/batch_count, acc, auc

    def valid_loop(self, data_loader):
        """
        The function to evaluate the model on the validation dataset.

        Parameters:
        - data_loader: PyTorch Dataloader for the validation dataset.

        Returns:
        - avg_loss: Average validation loss across all batches.
        - acc: Validation accuracy score.
        - auc: Validation area under the ROC curve score.
        """
        total_loss = 0
        gt_labels = []
        pred_labels = []
        pred_probs = None
        batch_count = len(data_loader)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (features_batch, label_batch) in enumerate(data_loader):
                # Store the ground truth labels
                if len(gt_labels) == 0:
                    gt_labels = label_batch.cpu().numpy().tolist()
                else:
                    gt_labels.extend(label_batch.cpu().numpy().tolist())

                # Move the input and label tensors to the GPU if specified
                features_batch = features_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                # Get the model's logits and predicted probabilities for the current batch
                logits, probs = self.model(features_batch)

                # Calculate the loss for the current batch
                loss = self.loss_fn(logits, label_batch)

                # Add the batch loss to the running total loss
                total_loss += loss.cpu().item()

                # Get the predicted labels and probabilities
                p_labels = torch.argmax(probs, dim=1)
                p_labels = p_labels.detach().cpu().numpy().tolist()
                p_probs = probs.detach().cpu().numpy()

                # Store the predicted labels and probabilities
                if len(pred_labels) == 0:
                    pred_probs = p_probs
                    pred_labels = p_labels
                else:
                    pred_probs = np.concatenate((pred_probs, p_probs), axis=0)
                    pred_labels.extend(p_labels)

                # Print the progress to the console
                sys.stdout.write('\rValidation Batch {}/{}, avg loss: {:.4f}'.format(
                    batch_idx + 1, batch_count, total_loss / (batch_idx + 1)))

            # Calculate the accuracy and AUC scores
            acc = accuracy_score(gt_labels, pred_labels)
            auc = roc_auc_score(gt_labels, pred_probs, multi_class='ovo',
                                labels=[i for i in range(self.num_classes)])
            # If verbose is set to 1, print the confusion matrix
            if self.verbose == 1:
                print('\nConfusion Matrix')
                conf = pd.DataFrame(confusion_matrix(gt_labels, pred_labels, labels=[i for i in range(self.num_classes)]),
                    index=['class_%d' % i for i in range(self.num_classes)],
                    columns=['class_%d' % i for i in range(self.num_classes)])
                print(conf)
        return total_loss/batch_count, acc, auc

    def test_loop(self, data_loader):
        """
        Evaluates the model on the given data_loader.

        Parameters:
        data_loader (torch.utils.data.DataLoader): The DataLoader that contains the test data.

        Returns:
        pred_labels (numpy.ndarray): The predicted labels for the test data.
        pred_probs (numpy.ndarray): The predicted probabilities for each class for the test data.

        """
        # Initialize the arrays for storing the predicted labels and probabilities
        pred_labels = None
        pred_probs = None

        # Get the number of batches in the data_loader
        batch_count = len(data_loader)

        # Put the model in evaluation mode
        self.model.eval()

        # Turn off gradient computation to speed up the evaluation
        with torch.no_grad():
            # Loop over the batches in the data_loader
            for batch_idx, (features_batch, _) in enumerate(data_loader):
                # Move the features batch to the device (GPU/CPU) specified in the instance
                features_batch = features_batch.to(self.device)

                # Forward the features through the model to get the logits and predicted probabilities
                logits, probs = self.model(features_batch)

                # Convert the predicted probabilities from tensor to numpy array
                probs = probs.detach().cpu().numpy()

                # Get the predicted labels as the class with the maximum predicted probability
                labels = np.argmax(probs, axis=1)

                # Concatenate the predicted labels and probabilities from this batch to the accumulated results
                if pred_labels is None:
                    pred_labels = labels
                    pred_probs = probs
                else:
                    pred_labels = np.concatenate((pred_labels, labels), axis=0)
                    pred_probs = np.concatenate((pred_probs, probs), axis=0)

                # Print the progress of the evaluation in the form of 'Batch x/y'
                sys.stdout.write('\rBatch {}/{}            '.format(batch_idx + 1, batch_count))

        # Return the accumulated predicted labels and probabilities
        return pred_labels, pred_probs
