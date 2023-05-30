import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer Perceptron (MLP) - Feedforward Neural Network Model

    This model is composed of multiple fully connected (Linear) layers followed by non-linear activation functions
    and dropout layers to reduce overfitting. The final layer is the output layer, which returns logits and class
    probabilities.

    Arguments:
        input_dim (int): number of input features
        hidden_dim (int): number of hidden units in each fully connected layer
        num_classes (int): number of classes in the classification task
        dropout (float): dropout rate applied after each fully connected layer

    """
    def __init__(self, input_dim=47, hidden_dim=512, num_classes=12, dropout=0.10):
        super(MLP, self).__init__()  # Inherited from the parent class nn.Module
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        """
        Forward Propagation of the MLP Model

        Arguments:
            batch (torch.Tensor): Input features of shape (batch_size, input_dim)

        Returns:
            logits (torch.Tensor): Logits of shape (batch_size, num_classes)
            probs (torch.Tensor): Class probabilities of shape (batch_size, num_classes)

        """
        features = self.fc(batch)
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs
