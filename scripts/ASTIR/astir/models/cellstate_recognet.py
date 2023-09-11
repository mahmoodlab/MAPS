"""
State Recognition Neural Network Model
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# The recognition net
class StateRecognitionNet(nn.Module):
    """State Recognition Neural Network to get mean of z and standard
    deviation of z. The neural network architecture looks like this: G ->
    const * C -> const * C -> G (for mu) or -> G (for std). With batch
    normal layers after each activation output layers and dropout
    activation units

    :param C: the number of pathways
    :param G: the number of proteins
    :param const: the size of the hidden layers are const times proportional
        to C, defaults to 2
    :param dropout_rate: the dropout rate, defaults to 0
    :param batch_norm: apply batch normal layers if True, defaults to False
    """

    def __init__(
        self,
        C: int,
        G: int,
        const: int = 2,
        dropout_rate: float = 0,
        batch_norm: bool = False,
    ) -> None:
        super(StateRecognitionNet, self).__init__()
        self.batch_norm = batch_norm

        hidden_layer_size = math.ceil(const * C)
        # First hidden layer
        self.linear1 = nn.Linear(G, hidden_layer_size).float()
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second hidden layer
        self.linear2 = nn.Linear(hidden_layer_size, hidden_layer_size).float()
        self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer for mu
        self.linear3_mu = nn.Linear(hidden_layer_size, C).float()
        self.dropout_mu = nn.Dropout(dropout_rate)

        # Output layer for std
        self.linear3_std = nn.Linear(hidden_layer_size, C).float()
        self.dropout_std = nn.Dropout(dropout_rate)

        # Batch normal layers
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(num_features=hidden_layer_size).float()
            self.bn2 = nn.BatchNorm1d(num_features=hidden_layer_size).float()
            self.bn_out_mu = nn.BatchNorm1d(num_features=C).float()
            self.bn_out_std = nn.BatchNorm1d(num_features=C).float()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One forward pass of the StateRecognitionNet

        :param x: the input to the recognition network model
        :return: the value from the output layer of the network
        """
        # Input --linear1--> Hidden1
        x = self.linear1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Hidden1 --linear2--> Hidden2
        x = self.linear2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Hidden2 --linear3_mu--> mu
        mu_z = self.linear3_mu(x)
        if self.batch_norm:
            mu_z = self.bn_out_mu(mu_z)
        mu_z = self.dropout_mu(mu_z)

        # Hidden2 --linear3_std--> std
        std_z = self.linear3_std(x)
        if self.batch_norm:
            std_z = self.bn_out_std(std_z)
        std_z = self.dropout_std(std_z)

        return mu_z, std_z
