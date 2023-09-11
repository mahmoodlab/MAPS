"""
Cell State Model
"""
import warnings
from collections import OrderedDict
from typing import Dict, Generator, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from astir.data import SCDataset

from .abstract import AstirModel
from .cellstate_recognet import StateRecognitionNet


class CellStateModel(AstirModel):
    """Class to perform statistical inference to on the activation
        of states (pathways) across cells

    :param dset: the input gene expression dataset, defaults to None
    :param const: See parameter ``const`` in
        :func:`astir.models.StateRecognitionNet`, defaults to 2
    :param dropout_rate: See parameter ``dropout_rate`` in
        :func:`astir.models.StateRecognitionNet`, defaults to 0
    :param batch_norm: See parameter ``batch_norm`` in
        :func:`astir.models.StateRecognitionNet`, defaults to False
    :param random_seed: the random seed number to reproduce results, defaults to 42
    :param dtype: torch datatype to use in the model, defaults to torch.float64
    :param device: torch.device's cpu or gpu, defaults to torch.device("cpu")
    """

    def __init__(
        self,
        dset: SCDataset = None,
        const: int = 2,
        dropout_rate: float = 0,
        batch_norm: bool = False,
        random_seed: int = 42,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(dset, random_seed, dtype, device)

        # Setting random seeds
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self._optimizer: Optional[torch.optim.Adam] = None
        self.const, self.dropout_rate, self.batch_norm = const, dropout_rate, batch_norm
        if self._dset is not None:
            self._param_init()

        # Convergence flag
        self._is_converged = False

    def _param_init(self) -> None:
        """Initializes sets of parameters"""
        if self._dset is None:
            raise Exception("the dataset is not provided")
        N = len(self._dset)
        C = self._dset.get_n_classes()
        G = self._dset.get_n_features()

        initializations = {
            "log_sigma": torch.log(self._dset.get_sigma().mean()),
            "mu": torch.reshape(self._dset.get_mu(), (1, -1)),
        }

        # Include beta or not
        d = torch.distributions.Uniform(
            torch.tensor(0.0, dtype=self._dtype), torch.tensor(1.5, dtype=self._dtype)
        )
        initializations["log_w"] = torch.log(d.sample((C, G)))

        self._variables = {
            n: i.to(self._device).detach().clone().requires_grad_()
            for (n, i) in initializations.items()
        }

        self._data = {
            "rho": self._dset.get_marker_mat().T.to(self._device),
        }

        self._recog = StateRecognitionNet(
            C,
            G,
            const=self.const,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
        ).to(device=self._device, dtype=self._dtype)

    def load_hdf5(self, hdf5_name: str) -> None:
        """Initializes Cell State Model from a hdf5 file type

        :param hdf5_name: file path
        """
        self._assignment = pd.read_hdf(
            hdf5_name, "cellstate_model/cellstate_assignments"
        )
        with h5py.File(hdf5_name, "r") as f:
            grp = f["cellstate_model"]
            param = grp["parameters"]
            self._variables = {
                "mu": torch.tensor(np.array(param["mu"])),
                "log_sigma": torch.tensor(np.array(param["log_sigma"])),
                "log_w": torch.tensor(np.array(param["log_w"])),
            }
            self._data = {"rho": torch.tensor(np.array(param["rho"]))}
            self._losses = torch.tensor(np.array(grp["losses"]["losses"]))

            rec = grp["recog_net"]
            hidden1_W = torch.tensor(np.array(rec["linear1.weight"]))
            hidden2_W = torch.tensor(np.array(rec["linear2.weight"]))
            hidden3_mu_W = torch.tensor(np.array(rec["linear3_mu.weight"]))
            hidden3_std_W = torch.tensor(np.array(rec["linear3_std.weight"]))
            state_dict = {
                "linear1.weight": hidden1_W,
                "linear1.bias": torch.tensor(np.array(rec["linear1.bias"])),
                "linear2.weight": hidden2_W,
                "linear2.bias": torch.tensor(np.array(rec["linear2.bias"])),
                "linear3_mu.weight": hidden3_mu_W,
                "linear3_mu.bias": torch.tensor(np.array(rec["linear3_mu.bias"])),
                "linear3_std.weight": hidden3_std_W,
                "linear3_std.bias": torch.tensor(np.array(rec["linear3_std.bias"])),
            }
            state_dict = OrderedDict(state_dict)
            self._recog = StateRecognitionNet(
                hidden3_mu_W.shape[0],
                hidden1_W.shape[1],
                const=self.const,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
            ).to(device=self._device, dtype=self._dtype)
            self._recog.load_state_dict(state_dict)
            self._recog.eval()

    def _loss_fn(
        self,
        mu_z: torch.Tensor,
        std_z: torch.Tensor,
        z_sample: torch.Tensor,
        y_in: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the calculated loss

        :param mu_z: the predicted mean of z
        :param std_z: the predicted standard deviation of z
        :param z_sample: the sampled z values
        :param y_in: the input datasets
        :return: the loss
        """
        S = y_in.shape[0]

        # log posterior q(z) approx p(z|y)
        q_z_dist = torch.distributions.Normal(loc=mu_z, scale=torch.exp(std_z))
        log_q_z = q_z_dist.log_prob(z_sample)

        # log likelihood p(y|z)
        rho_w = torch.mul(self._data["rho"], torch.exp(self._variables["log_w"]))
        mean = self._variables["mu"] + torch.matmul(z_sample, rho_w)
        std = torch.exp(self._variables["log_sigma"]).reshape(1, -1)
        p_y_given_z_dist = torch.distributions.Normal(loc=mean, scale=std)
        log_p_y_given_z = p_y_given_z_dist.log_prob(y_in)

        # log prior p(z)
        p_z_dist = torch.distributions.Normal(0, 1)
        log_p_z = p_z_dist.log_prob(z_sample)

        loss = (1 / S) * (
            torch.sum(log_q_z) - torch.sum(log_p_y_given_z) - torch.sum(log_p_z)
        )

        return loss

    def _forward(
        self,
        Y: Optional[torch.Tensor],
        X: Optional[torch.Tensor] = None,
        design: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One forward pass

        :param Y: dataset to do forward pass on
        :return: mu_z, std_z, z_sample
        """
        mu_z, std_z = self._recog(Y)

        std = torch.exp(std_z)
        eps = torch.randn_like(std)
        z_sample = eps * std + mu_z

        return mu_z, std_z, z_sample

    def fit(
        self,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        delta_loss: float = 1e-3,
        delta_loss_batch: int = 10,
        msg: str = "",
    ) -> None:
        """ Runs train loops until the convergence reaches delta_loss for\
            delta_loss_batch sizes or for max_epochs number of times

        :param max_epochs: number of train loop iterations, defaults to 50
        :param learning_rate: the learning rate, defaults to 0.01
        :param batch_size: the batch size, defaults to 128
        :param delta_loss: stops iteration once the loss rate reaches\
            delta_loss, defaults to 0.001
        :param delta_loss_batch: the batch size to consider delta loss,\
            defaults to 10
        :param msg: iterator bar message, defaults to empty string
        """
        if self._dset is None:
            raise Exception("the dataset is not provided")

        # Returns early if the model has already converged
        if self._is_converged:
            return

        # Create an optimizer if there is no optimizer
        if self._optimizer is None:
            opt_params = list(self._recog.parameters())
            opt_params += list(self._variables.values())  # type: ignore
            self._optimizer = torch.optim.Adam(opt_params, lr=learning_rate)

        iterator = trange(
            max_epochs,
            desc="training restart" + msg,
            unit="epochs",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
        )
        train_iterator = DataLoader(
            self._dset, batch_size=min(batch_size, len(self._dset))
        )
        for ep in iterator:
            for i, (y_in, x_in, _) in enumerate(train_iterator):
                self._optimizer.zero_grad()

                mu_z, std_z, z_samples = self._forward(
                    x_in.type(self._dtype).to(self._device)
                )

                loss = self._loss_fn(
                    mu_z, std_z, z_samples, x_in.type(self._dtype).to(self._device)
                )

                loss.backward()

                self._optimizer.step()

            loss_detached = loss.cpu().detach().item()

            self._losses = torch.cat(
                (self._losses, torch.tensor([loss_detached], dtype=self._dtype))
            )

            if len(self._losses) > delta_loss_batch:
                curr_mean = torch.mean(self._losses[-delta_loss_batch:])
                prev_mean = torch.mean(self._losses[-delta_loss_batch - 1 : -1])
                curr_delta_loss = (prev_mean - curr_mean) / prev_mean
                delta_cond_met = 0 <= curr_delta_loss.item() < delta_loss
            else:
                delta_cond_met = False

            iterator.set_postfix_str("current loss: " + str(round(loss_detached, 1)))

            if delta_cond_met:
                self._is_converged = True
                iterator.close()
                break

        g = self.get_final_mu_z().detach().cpu().numpy()
        self._assignment = pd.DataFrame(g)
        self._assignment.columns = self._dset.get_classes()
        self._assignment.index = self._dset.get_cell_names()

    def get_recognet(self) -> StateRecognitionNet:
        """Getter for the recognition net

        :return: the recognition net
        """
        return self._recog

    def get_final_mu_z(self, new_dset: Optional[SCDataset] = None) -> torch.Tensor:
        """Returns the mean of the predicted z values for each core

        :param new_dset: returns the predicted z values of this dataset on
            the existing model. If None, it predicts using the existing
            dataset, defaults to None
        :return: the mean of the predicted z values for each core
        """
        if self._dset is None:
            raise Exception("the dataset is not provided")
        if new_dset is None:
            _, x_in, _ = self._dset[:]  # should be the scaled
            # one
        else:
            _, x_in, _ = new_dset[:]
        final_mu_z, _, _ = self._forward(x_in.type(self._dtype).to(self._device))

        return final_mu_z

    def get_correlations(self) -> np.array:
        """Returns a C (# of pathways) X G (# of proteins) matrix
        where each element represents the correlation value of the pathway
        and the protein

        :return: matrix of correlation between all pathway and protein pairs.
        """
        if self._dset is None:
            raise Exception("No dataset input to the model")

        state_assignment = self.get_final_mu_z().detach().cpu().numpy()
        y_in = self._dset.get_exprs()

        feature_names = self._dset.get_features()
        state_names = self._dset.get_classes()
        G = self._dset.get_n_features()
        C = self._dset.get_n_classes()
        corr_mat = np.zeros((C, G))
        # Make a matrix of correlations between all states and proteins
        for c, state in enumerate(state_names):
            for g, feature in enumerate(feature_names):
                states = state_assignment[:, c]
                protein = y_in[:, g].cpu()
                corr_mat[c, g] = np.corrcoef(protein, states)[0, 1]

        return corr_mat

    def diagnostics(self) -> pd.DataFrame:
        """Run diagnostics on cell type assignments

        See :meth:`astir.Astir.diagnostics_cellstate` for full documentation
        """
        if self._dset is None:
            raise Exception("the dataset is not provided")
        feature_names = self._dset.get_features()
        state_names = self._dset.get_classes()

        corr_mat = self.get_correlations()

        # Correlation values of all marker proteins
        marker_mat = self._dset.get_marker_mat().T.cpu().numpy()
        marker_corr = marker_mat * corr_mat
        marker_corr[marker_mat == 0] = np.inf

        # Smallest correlation values for each pathway
        min_marker_corr = np.min(marker_corr, axis=1).reshape(-1, 1)
        min_marker_proteins = np.take(feature_names, np.argmin(marker_corr, axis=1))

        # Correlation values of all non marker proteins
        non_marker_mat = 1 - self._dset.get_marker_mat().T.cpu().numpy()
        non_marker_corr = non_marker_mat * corr_mat
        non_marker_corr[non_marker_mat == 0] = -np.inf

        # Any correlation values where non marker proteins is greater than
        # the smallest correlation values of marker proteins
        bad_corr_marker = np.array(non_marker_corr > min_marker_corr, dtype=np.int32)

        # Problem summary
        indices = np.argwhere(bad_corr_marker > 0)

        col_names = [
            "pathway",
            "protein A",
            "correlation of protein A",
            "protein B",
            "correlation of protein B",
            "note",
        ]

        problems = []
        for index in indices:
            state_index = index[0]
            protein_index = index[1]
            state = state_names[index[0]]
            marker_protein = min_marker_proteins[state_index]
            non_marker_protein = feature_names[protein_index]
            problem = {
                "pathway": state,
                "marker_protein": marker_protein,
                "corr_of_marker_protein": min_marker_corr[state_index][0],
                "non_marker_protein": non_marker_protein,
                "corr_of_non_marker_protein": non_marker_corr[
                    state_index, protein_index
                ],
                "msg": "{} is marker for {} but {} isn't".format(
                    marker_protein, state, non_marker_protein
                ),
            }
            problems.append(problem)

        if len(problems) > 0:
            df_issues = pd.DataFrame(problems)
            df_issues.columns = col_names
        else:
            df_issues = pd.DataFrame(columns=col_names)

        return df_issues


class NotClassifiableError(RuntimeError):
    """Raised when the input datasets is not classifiable."""

    pass
