import torch
import torch.nn as nn
import torch.nn.functional as F


class TypeRecognitionNet(nn.Module):
    """Type Recognition Neural Network.

    :param C: number of classes
    :param G: number of features
    :param hidden_size: size of hidden layers, defaults to 10
    """

    def __init__(self, C: int, G: int, hidden_size: int = 20) -> None:
        super(TypeRecognitionNet, self).__init__()
        self.hidden_1 = nn.Linear(G, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, C + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One forward pass.

        :param x: the input vector
        :return: the calculated cost value
        """
        x = self.hidden_1(x)
        x = F.leaky_relu(x)
        x = self.hidden_2(x)

        return F.softmax(x, dim=1), F.log_softmax(x, dim=1)
