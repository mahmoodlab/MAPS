""" 
Cell Type Model
"""

import re
import warnings
from collections import OrderedDict
from typing import Dict, Generator, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from torch.distributions import (
    LowRankMultivariateNormal,
    MultivariateNormal,
    Normal,
    StudentT,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from astir.data import SCDataset

from .abstract import AstirModel
from .celltype_recognet import TypeRecognitionNet


class CellTypeModel(AstirModel):
    """Class to perform statistical inference to assign cells to cell types.

    :param dset: the input gene expression dataframe
    :param random_seed: the random seed for parameter initialization, defaults to 1234
    :param dtype: the datasets type of parameters, should be the same as `dset`, defaults to
        torch.float64
    """

    def __init__(
        self,
        dset: Optional[SCDataset] = None,
        random_seed: int = 1234,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(dset, random_seed, dtype, device)

        if dset is not None:
            self._param_init()

    def _param_init(self) -> None:
        """Initializes parameters and design matrices."""
        if self._dset is None:
            raise Exception("the dataset is not provided")
        G = self._dset.get_n_features()
        C = self._dset.get_n_classes()

        self._recog = TypeRecognitionNet(
            self._dset.get_n_classes(), self._dset.get_n_features()
        ).to(self._device, dtype=self._dtype)

        # Establish datasets
        self._data: Dict[str, torch.Tensor] = {
            # "log_alpha": torch.log(torch.ones(C + 1, dtype=self._dtype) / (C + 1)).to(
            #     self._device
            # ),
            "rho": self._dset.get_marker_mat().to(self._device),
        }

        self._alpha_prior = Dirichlet(
            torch.ones(C + 1, dtype=self._dtype).to(self._device) * (C + 1)
        )

        # Initialize mu, log_delta
        delta_init_mean = torch.log(
            torch.log(torch.tensor(3.0, dtype=self._dtype))
        )  # the log of the log of this is the multiplier
        t = torch.distributions.Normal(
            # delta_init_mean.clone().detach().to(self._dtype),
            torch.tensor(0, dtype=self._dtype),
            torch.tensor(0.1, dtype=self._dtype),
        )
        log_delta_init = t.sample((G, C + 1))

        mu_init = torch.log(
            torch.tensor(self._dset.get_mu_init(), dtype=self._dtype)
        ).to(self._device)
        # mu_init = torch.log(self._dset.get_mu()).to(self._device)
        # mu_init = mu_init - (
        #     self._data["rho"] * torch.exp(log_delta_init).to(self._device)
        # ).mean(1)

        mu_init = mu_init.reshape(-1, 1)

        # Create initialization dictionary
        initializations = {
            "mu": mu_init,
            "log_sigma": torch.log(self._dset.get_sigma()).to(self._device),
            "log_delta": log_delta_init,
            "p": torch.zeros((G, C + 1), dtype=self._dtype, device=self._device),
            "alpha_logits": torch.ones(C + 1, dtype=self._dtype, device=self._device),
        }
        P = self._dset.get_design().shape[1]
        # Add additional columns of mu for anything in the design matrix
        initializations["mu"] = torch.cat(
            [
                initializations["mu"],
                torch.zeros((G, P - 1), dtype=self._dtype, device=self._device),
            ],
            1,
        )
        # Create trainable variables
        self._variables: Dict[str, torch.Tensor] = {}
        for (n, v) in initializations.items():
            self._variables[n] = Variable(v.clone()).to(self._device)
            self._variables[n].requires_grad = True

    def load_hdf5(self, hdf5_name: str) -> None:
        """Initializes Cell Type Model from a hdf5 file type

        :param hdf5_name: file path
        """
        self._assignment = pd.read_hdf(hdf5_name, "celltype_model/celltype_assignments")
        with h5py.File(hdf5_name, "r") as f:
            grp = f["celltype_model"]
            param = grp["parameters"]
            self._variables = {
                "mu": torch.tensor(np.array(param["mu"])),
                "log_sigma": torch.tensor(np.array(param["log_sigma"])),
                "log_delta": torch.tensor(np.array(param["log_delta"])),
                "p": torch.tensor(np.array(param["p"])),
                "alpha_logits": torch.tensor(np.array(param["alpha_logits"])),
            }
            self._data = {
                "rho": torch.tensor(np.array(param["rho"])),
            }
            self._losses = torch.tensor(np.array(grp["losses"]["losses"]))

            rec = grp["recog_net"]
            hidden1_W = torch.tensor(np.array(rec["hidden_1.weight"]))
            hidden2_W = torch.tensor(np.array(rec["hidden_2.weight"]))
            state_dict = {
                "hidden_1.weight": hidden1_W,
                "hidden_1.bias": torch.tensor(np.array(rec["hidden_1.bias"])),
                "hidden_2.weight": hidden2_W,
                "hidden_2.bias": torch.tensor(np.array(rec["hidden_2.bias"])),
            }
            state_dict = OrderedDict(state_dict)
            self._recog = TypeRecognitionNet(
                hidden2_W.shape[0] - 1, hidden1_W.shape[1], hidden1_W.shape[0]
            ).to(device=self._device, dtype=self._dtype)
            self._recog.load_state_dict(state_dict)
            self._recog.eval()

    def _forward(
        self, Y: torch.Tensor, X: torch.Tensor, design: torch.Tensor
    ) -> torch.Tensor:
        """One forward pass.

        :param Y: a sample from the dataset
        :param X: normalized sample datasets
        :param design: the corresponding row of design matrix
        :return: the cost (elbo) of the current pass
        """
        if self._dset is None:
            raise Exception("the dataset is not provided")
        G = self._dset.get_n_features()
        C = self._dset.get_n_classes()
        N = Y.shape[0]

        Y_spread = Y.view(N, 1, G).repeat(1, C + 1, 1)

        delta_tilde = torch.exp(self._variables["log_delta"])
        mean = delta_tilde * self._data["rho"]
        mean2 = torch.mm(design, self._variables["mu"].T)  ## N x P * P x G
        mean2 = mean2.view(-1, G, 1).repeat(1, 1, C + 1)
        mean = mean + mean2

        # now do the variance modelling
        p = torch.sigmoid(self._variables["p"])

        sigma = torch.exp(self._variables["log_sigma"])
        v1 = (self._data["rho"] * p).T * sigma
        v2 = torch.pow(sigma, 2) * (1 - torch.pow(self._data["rho"] * p, 2)).T

        v1 = v1.view(1, C + 1, G, 1).repeat(N, 1, 1, 1)  # extra 1 is the "rank"
        v2 = v2.view(1, C + 1, G).repeat(N, 1, 1) + 1e-6

        dist = LowRankMultivariateNormal(
            loc=torch.exp(mean).permute(0, 2, 1), cov_factor=v1, cov_diag=v2
        )

        log_p_y_on_c = dist.log_prob(Y_spread)

        gamma, log_gamma = self._recog.forward(X)
        log_alpha = F.log_softmax(self._variables["alpha_logits"], dim=0)
        alpha = F.softmax(self._variables["alpha_logits"], dim=0)
        mix_prior = self._alpha_prior.log_prob(alpha)

        elbo = (gamma * (log_p_y_on_c + log_alpha - log_gamma)).sum() + mix_prior

        return -elbo

    def fit(
        self,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        delta_loss: float = 1e-3,
        delta_loss_batch: int = 10,
        msg: str = "",
    ) -> None:
        """Runs train loops until the convergence reaches delta_loss for
        delta_loss_batch sizes or for max_epochs number of times

        :param max_epochs: number of train loop iterations, defaults to 50
        :param learning_rate: the learning rate, defaults to 0.01
        :param batch_size: the batch size, defaults to 128
        :param delta_loss: stops iteration once the loss rate reaches
            delta_loss, defaults to 0.001
        :param delta_loss_batch: the batch size to consider delta loss,
            defaults to 10
        :param msg: iterator bar message, defaults to empty string
        """
        if self._dset is None:
            raise Exception("the dataset is not provided")
        # Make dataloader
        dataloader = DataLoader(
            self._dset, batch_size=min(batch_size, len(self._dset)), shuffle=True
        )

        # Run training loop
        losses: List[torch.Tensor] = []
        per = torch.tensor(1)

        # Construct optimizer
        opt_params = list(self._variables.values()) + list(self._recog.parameters())

        optimizer = torch.optim.Adam(opt_params, lr=learning_rate)

        _, exprs_X, _ = self._dset[:]  # calls dset.get_item

        iterator = trange(
            max_epochs,
            desc="training restart" + msg,
            unit="epochs",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
        )
        for ep in iterator:
            # for ep in range(max_epochs):
            L = None
            loss = torch.tensor(0.0, dtype=self._dtype)
            for batch in dataloader:
                Y, X, design = batch
                optimizer.zero_grad()
                L = self._forward(Y, X, design)
                L.backward()
                optimizer.step()
                with torch.no_grad():
                    loss = loss + L
            if len(losses) > 0:
                per = abs((loss - losses[-1]) / losses[-1])
            losses.append(loss)
            iterator.set_postfix_str("current loss: " + str(round(float(loss), 1)))

            if per <= delta_loss:
                self._is_converged = True
                iterator.close()
                break

        # Save output
        self._assignment = pd.DataFrame(
            self._recog.forward(exprs_X)[0].detach().cpu().numpy()
        )
        self._assignment.columns = self._dset.get_classes() + ["Other"]
        self._assignment.index = self._dset.get_cell_names()

        if self._losses.shape[0] == 0:
            self._losses = torch.tensor(losses)
        else:
            self._losses = torch.cat(
                (self._losses.view(self._losses.shape[0]), torch.tensor(losses)), dim=0
            )

    def predict(self, new_dset: pd.DataFrame) -> np.array:
        """Feed `new_dset` to the recognition net to get a prediction.

        :param new_dset: the dataset to be predicted
        :return: the resulting cell type assignment
        """
        # exprs_X = torch.tensor(new_dset[:].values, device=self._device)
        _, exprs_X, _ = self._dset[:]
        g = pd.DataFrame(self._recog.forward(exprs_X)[0].detach().cpu().numpy())
        return g

    def get_recognet(self) -> TypeRecognitionNet:
        """Getter for the recognition net.

        :return: the trained recognition net
        """
        return self._recog

    def _most_likely_celltype(
        self,
        row: pd.DataFrame,
        threshold: float,
        cell_types: List[str],
        assignment_type: str,
    ) -> str:
        """Given a row of the assignment matrix, return the most likely cell type

        :param row: the row of cell assignment matrix to be evaluated
        :param threshold: the higher bound of the maximun probability to classify a cell as `Unknown`
        :param cell_types: the names of cell types, in the same order as the features of the row
        :param assignment_type: See
        :meth:`astir.CellTypeModel.get_celltypes` for full documentation
        :return: the most likely cell type of this cell
        """
        row = row.values
        max_prob = np.max(row)

        if assignment_type == "threshold":
            if max_prob < threshold:
                return "Unknown"
        elif assignment_type == "max":
            if sum(row == max_prob) > 1:
                return "Unknown"

        return cell_types[np.argmax(row)]

    def get_celltypes(
        self,
        threshold: float = 0.7,
        assignment_type: str = "threshold",
        prob_assign: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Get the most likely cell types. A cell is assigned to a cell type
        if the probability is greater than threshold.
        If no cell types have a probability higher than threshold,
        then "Unknown" is returned.

        :param assignment_type: either 'threshold' or 'max'. If threshold,
            type assignment is based on whether the probability threshold is
            above prob_assignment. If 'max', type assignment is based on the max
            probability value or "unknown" if there are multiple max
            probabilities. Defaults to 'threshold'.
        :param threshold: the probability threshold above which a cell is
            assigned to a cell type, defaults to 0.7
        :return: a datasets frame with most likely cell types for each
        """
        if prob_assign is None:
            type_probability = self.get_assignment()
        else:
            type_probability = prob_assign

        if assignment_type != "threshold" and assignment_type != "max":
            warnings.warn(
                "Wrong assignment type. Defaults the assignment " "type to threshold."
            )
            assignment_type = "threshold"

        if assignment_type == "max" and prob_assign is not None:
            warnings.warn(
                "Assignment type is 'max' but probability "
                "threshold value was passed in. Probability "
                "threshold value will be ignored."
            )

        cell_types = list(type_probability.columns)

        cell_type_assignments = type_probability.apply(
            self._most_likely_celltype,
            axis=1,
            assignment_type=assignment_type,
            threshold=threshold,
            cell_types=cell_types,
        )
        cell_type_assignments = pd.DataFrame(cell_type_assignments)
        cell_type_assignments.columns = ["cell_type"]

        return cell_type_assignments

    def _compare_marker_between_types(
        self,
        curr_type: str,
        celltype_to_compare: str,
        marker: str,
        cell_types: List[str],
        alpha: float = 0.05,
    ) -> Optional[dict]:
        """For two cell types and a protein, ensure marker
        is expressed at higher level for curr_type than celltype_to_compare

        :param curr_type: the cell type to assess
        :param celltype_to_compare: all the cell types that shouldn't highly express this marker
        :param marker: the marker protein for curr_type
        :param cell_types: list of cell types assigned for cells
        :param alpha:
        :return:
        """
        if self._dset is None:
            raise Exception("the dataset is not provided")
        current_marker_ind = np.array(self._dset.get_features()) == marker

        cells_x = np.array(cell_types) == curr_type
        cells_y = np.array(cell_types) == celltype_to_compare

        # x - cells whose cell types' marker protein is marker
        # y - cells whose cell types' marker protein is not marker
        x = self._dset.get_exprs().detach().cpu().numpy()[cells_x, current_marker_ind]
        y = self._dset.get_exprs().detach().cpu().numpy()[cells_y, current_marker_ind]

        stat = np.NaN
        pval = np.Inf
        note: Optional[str] = "Only 1 cell in a type: comparison not possible"

        if len(x) > 1 and len(y) > 1:
            tt = stats.ttest_ind(x, y)
            stat = tt.statistic
            pval = tt.pvalue
            note = None

        if not (stat > 0 and pval < alpha):
            rdict = {
                "current_marker": marker,
                "curr_type": curr_type,
                "celltype_to_compare": celltype_to_compare,
                "mean_A": x.mean(),
                "mean_Y": y.mean(),
                "p-val": pval,
                "note": note,
            }

            return rdict

        return None

    def plot_clustermap(
        self,
        plot_name: str = "celltype_protein_cluster.png",
        threshold: float = 0.7,
        figsize: Tuple[float, float] = (7.0, 5.0),
        prob_assign: Optional[pd.DataFrame] = None,
    ) -> None:
        """Save the heatmap of protein content in cells with cell types labeled.

        :param plot_name: name of the plot, extension(e.g. .png or .jpg) is needed, defaults to "celltype_protein_cluster.png"
        :param threshold: the probability threshold above which a cell is assigned to a cell type, defaults to 0.7
        :param figsize: the size of the figure, defaults to (7.0, 5.0)
        """
        if self._dset is None:
            raise Exception("the dataset is not provided")
        expr_df = self._dset.get_exprs_df()
        scaler = StandardScaler()
        for feature in expr_df.columns:
            expr_df[feature] = scaler.fit_transform(
                expr_df[feature].values.reshape((expr_df[feature].shape[0], 1))
            )

        expr_df["cell_type"] = self.get_celltypes(
            threshold=threshold, prob_assign=prob_assign
        )
        expr_df = expr_df.sort_values(by=["cell_type"])
        types = expr_df.pop("cell_type")
        types_uni = types.unique()

        lut = dict(zip(types_uni, sns.color_palette("BrBG", len(types_uni))))
        col_colors = pd.DataFrame(types.map(lut))
        cm = sns.clustermap(
            expr_df.T,
            xticklabels=False,
            cmap="vlag",
            col_cluster=False,
            col_colors=col_colors,
            figsize=figsize,
        )

        for t in types_uni:
            cm.ax_col_dendrogram.bar(0, 0, color=lut[t], label=t, linewidth=0)
        cm.ax_col_dendrogram.legend(
            title="Cell Types", loc="center", ncol=3, bbox_to_anchor=(0.8, 0.8)
        )
        cm.savefig(plot_name, dpi=150)

    def diagnostics(self, cell_type_assignments: list, alpha: float) -> pd.DataFrame:
        """Run diagnostics on cell type assignments

        See :meth:`astir.Astir.diagnostics_celltype` for full documentation
        """
        if self._dset is None:
            raise Exception("the dataset is not provided")
        problems = []

        # Want to construct a datasets frame that models rho with
        # cell type names on the columns and feature names on the rows
        g_df = pd.DataFrame(self._data["rho"].detach().cpu().numpy())
        g_df.columns = self._dset.get_classes() + ["Other"]
        g_df.index = self._dset.get_features()

        for curr_type in self._dset.get_classes():
            if not curr_type in cell_type_assignments:
                continue

            current_markers = g_df.index[g_df[curr_type] == 1]

            for current_marker in current_markers:
                # find all the cell types that shouldn't highly express this marker
                celltypes_to_compare = g_df.columns[g_df.loc[current_marker] == 0]

                for celltype_to_compare in celltypes_to_compare:
                    if not celltype_to_compare in cell_type_assignments:
                        continue

                    is_problem = self._compare_marker_between_types(
                        curr_type,
                        celltype_to_compare,
                        current_marker,
                        cell_type_assignments,
                        alpha,
                    )

                    if is_problem is not None:
                        problems.append(is_problem)

        col_names = [
            "feature",
            "should be expressed higher in",
            "than",
            "mean cell type 1",
            "mean cell type 2",
            "p-value",
            "note",
        ]
        df_issues = None
        if len(problems) > 0:
            df_issues = pd.DataFrame(problems)
            df_issues.columns = col_names
        else:
            df_issues = pd.DataFrame(columns=col_names)

        return df_issues
