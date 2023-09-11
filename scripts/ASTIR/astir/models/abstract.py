import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from astir.data import SCDataset


class AstirModel:
    """Abstract class to perform statistical inference to assign. This module is the super class of
    `CellTypeModel` and `CellStateModel` and is not supposed to be instantiated.
    """

    def __init__(
        self,
        dset: Optional[SCDataset],
        random_seed: int,
        dtype: torch.dtype,
        device: torch.device = torch.device("cpu"),
    ) -> None:

        if not isinstance(random_seed, int):
            raise NotClassifiableError("Random seed is expected to be an integer.")
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        if dtype != torch.float32 and dtype != torch.float64:
            raise NotClassifiableError(
                "dtype must be one of torch.float32 and torch.float64."
            )
        elif dset is not None and dtype != dset.get_dtype():
            raise NotClassifiableError("dtype must be the same as `dset`.")
        self._dtype: torch.dtype = dtype
        self._data: Dict[str, torch.Tensor] = {}
        self._variables: Dict[str, torch.Tensor] = {}
        self._losses: torch.Tensor = torch.tensor([], dtype=self._dtype)
        self._assignment: pd.DataFrame = pd.DataFrame()

        self._dset = dset
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._is_converged = False

    def get_losses(self) -> torch.Tensor:
        """Getter for losses.

        :return: self.losses
        """
        if len(self._losses) == 0:
            raise Exception("The model has not been trained yet")
        return self._losses

    def get_scdataset(self) -> SCDataset:
        """Getter for the `SCDataset`.

        :return: `self._dset`
        """
        if self._dset is None:
            raise Exception("the dataset is not provided")
        return self._dset

    def get_data(self) -> Dict[str, torch.Tensor]:
        """Get model datasets

        :return: datasets
        """
        if self._data == {}:
            raise Exception("The model has not been initialized yet")
        return self._data

    def get_variables(self) -> Dict[str, torch.Tensor]:
        """Returns all variables

        :return: self._variables
        """
        if self._variables == {}:
            raise Exception("The model has not been initialized yet")
        return self._variables

    def is_converged(self) -> bool:
        """Returns True if the model converged

        :return: self._is_converged
        """
        return self._is_converged

    def get_assignment(self) -> pd.DataFrame:
        """Get the final assignment of the dataset.

        :return: the final assignment of the dataset
        """
        if self._assignment.shape == (0, 0):
            raise Exception("The model has not been trained yet")
        return self._assignment

    def _param_init(self) -> None:
        """Initializes parameters and design matrices."""
        raise NotImplementedError("AbstractModel is not supposed to be instantiated.")

    def _forward(
        self, Y: torch.Tensor, X: torch.Tensor, design: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """One forward pass"""
        raise NotImplementedError("AbstractModel is not supposed to be instantiated.")

    def fit(
        self,
        max_epochs: int,
        learning_rate: float,
        batch_size: int,
        delta_loss: float,
        delta_loss_batch: int,
        msg: str,
    ) -> None:
        """Runs train loops until the convergence reaches delta_loss for
        delta_loss_batch sizes or for max_epochs number of times
        """
        raise NotImplementedError("AbstractModel is not supposed to be instantiated.")


class NotClassifiableError(RuntimeError):
    """Raised when the input datasets is not classifiable."""

    pass
