# astir: Automated aSsignmenT sIngle-cell pRoteomics

# VSCode tips for python:
# ctrl+` to open terminal
# cmd+P to go to file
import os
import re
import warnings
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .data.scdataset import SCDataset
from .models.cellstate import CellStateModel
from .models.celltype import CellTypeModel


class Astir:
    """Create an Astir object

    :param input_expr: the single cell protein expression dataset
    :type input_expr: Union[pd.DataFrame, Tuple[np.array, List[str], List[str]], Tuple[SCDataset, SCDataset]]
    :param marker_dict: the marker dictionary which maps cell type/state to protein features, defaults to None
    :type marker_dict: Dict[str, Dict[str, str]], optional
    :param design: the design matrix labeling the grouping of cell, defaults to None
    :type design: Union[pd.DataFrame, np.array], optional
    :param random_seed: random seed for parameter initialization, defaults to `1234`
    :type random_seed: int, optional
    :param dtype: dtype of datasets, defaults to `torch.float64`
    :type dtype: torch.dtype, optional
    :raises NotClassifiableError: raised if the model is not trainable
    """

    def __init__(
        self,
        input_expr: Union[
            pd.DataFrame,
            Tuple[np.array, List[str], List[str]],
            Tuple[SCDataset, SCDataset],
        ] = None,
        marker_dict: Optional[Dict[str, Dict[str, List[str]]]] = None,
        design: Union[pd.DataFrame, np.array, None] = None,
        random_seed: int = 1234,
        dtype: torch.dtype = torch.float64,
    ) -> None:

        if not isinstance(random_seed, int):
            raise NotClassifiableError("Random seed is expected to be an integer.")
        torch.manual_seed(random_seed)
        self.random_seed = random_seed

        if dtype != torch.float32 and dtype != torch.float64:
            raise NotClassifiableError(
                "Dtype must be one of torch.float32 and torch.float64."
            )
        self._dtype = dtype

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._type_ast: Optional[CellTypeModel] = None
        self._state_ast: Optional[CellStateModel] = None
        self._type_run_info: dict = {}
        self._state_run_info: dict = {}

        self._type_dset: Optional[SCDataset] = None
        self._state_dset: Optional[SCDataset] = None

        self._hierarchy_dict: Optional[Dict[str, List[str]]] = None

        type_dict, state_dict, self._hierarchy_dict = self._sanitize_dict(marker_dict)
        if isinstance(input_expr, tuple) and len(input_expr) == 2:
            self._type_dset, self._state_dset = input_expr[0], input_expr[1]
        else:
            if type_dict is not None:
                self._type_dset = SCDataset(
                    expr_input=input_expr,
                    marker_dict=type_dict,
                    design=design,
                    include_other_column=True,
                    dtype=self._dtype,
                    device=self._device,
                )
            if state_dict is not None:
                self._state_dset = SCDataset(
                    expr_input=input_expr,
                    marker_dict=state_dict,
                    design=design,
                    include_other_column=False,
                    dtype=self._dtype,
                    device=self._device,
                )

    def _sanitize_dict(
        self, marker_dict: Optional[Dict[str, Dict[str, List[str]]]]
    ) -> Union[List[None], List[dict]]:
        """Sanitizes the marker dictionary.

        :param marker_dict: dictionary read from the yaml file
        :raises NotClassifiableError: raized when the marker dictionary doesn't
             have required format
        :return: dictionaries, the first is the cell type dict, the second is the cell state
            dict and the third is the hierarchy dict.
        """
        dics = [None, None, None]
        if marker_dict is not None:
            ct = re.compile("cell[^a-zA-Z0-9]*type", re.IGNORECASE)
            cs = re.compile("cell[^a-zA-Z0-9]*state", re.IGNORECASE)
            h = re.compile(".*hierarch.*", re.IGNORECASE)

            def interpret(key: str, dic: Any) -> None:
                if ct.match(key):
                    dics[0] = dic
                elif cs.match(key):
                    dics[1] = dic
                elif h.match(key):
                    dics[2] = dic

            for key, dic in marker_dict.items():
                interpret(key, dic)

        return dics

    def fit_type(
        self,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        delta_loss: float = 1e-3,
        n_init: int = 5,
        n_init_epochs: int = 5,
    ) -> None:
        """Run Variational Bayes to infer cell types

        :param max_epochs: maximum number of epochs to train
        :param learning_rate: ADAM optimizer learning rate
        :param batch_size: minibatch size
        :param delta_loss: stops iteration once the loss rate reaches
            delta_loss, defaults to `0.001`
        :param n_inits: number of random initializations
        """
        if self._type_dset is None:
            raise NotClassifiableError(
                "Dataset or marker for cell type classification is not provided"
            )
        np.random.seed(self.random_seed)
        seeds = np.random.randint(1, 100000000, n_init)
        type_models = [
            CellTypeModel(
                self._type_dset,
                int(seed),
                self._dtype,
                self._device,
            )
            for seed in seeds
        ]
        n_init_epochs = min(max_epochs, n_init_epochs)
        for i in range(n_init):
            type_models[i].fit(
                n_init_epochs,
                learning_rate,
                batch_size,
                delta_loss,
                msg=" " + str(i + 1) + "/" + str(n_init),
            )

        losses = torch.tensor([m.get_losses()[-1] for m in type_models])

        best_ind = int(torch.argmin(losses).item())
        self._type_ast = type_models[best_ind]
        self._type_ast.fit(
            max_epochs, learning_rate, batch_size, delta_loss, msg=" (final)"
        )

        if not self._type_ast.is_converged():
            msg = (
                "Maximum epochs reached. More iteration may be needed to"
                + " complete the training."
            )
            warnings.warn(msg)

        self._type_run_info = {
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "delta_loss": delta_loss,
            "n_init": n_init,
            "n_init_epochs": n_init_epochs,
        }
        return None

    def fit_state(
        self,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        delta_loss: float = 1e-3,
        n_init: int = 5,
        n_init_epochs: int = 5,
        delta_loss_batch: int = 10,
        const: int = 2,
        dropout_rate: float = 0,
        batch_norm: bool = False,
    ) -> None:
        """Run Variational Bayes to infer cell states

        :param max_epochs: number of epochs, defaults to `100`
        :param learning_rate: the learning rate, defaults to `1e-2`
        :param n_init: the number of initial parameters to compare,
            defaults to `5`
        :param delta_loss: stops iteration once the loss rate reaches
            delta_loss, defaults to `0.001`
        :param delta_loss_batch: the batch size to consider delta loss,
            defaults to `10`
        """
        if self._state_dset is None:
            raise NotClassifiableError(
                "Dataset or marker for cell state classification is not provided"
            )
        cellstate_models = []
        cellstate_losses = []

        if delta_loss_batch >= max_epochs:
            warnings.warn("Delta loss batch size is greater than the number of epochs")

        if n_init_epochs > max_epochs:
            warnings.warn(
                "n_init_epochs cannot be greater than the "
                "max_epochs. Models will be trained with max_epochs."
            )

        for i in range(n_init):
            # Initializing a model
            model = CellStateModel(
                const=const,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                dset=self._state_dset,
                random_seed=(self.random_seed + i),
                dtype=self._dtype,
            )
            # Fitting the model
            n_init_epochs = min(max_epochs, n_init_epochs)
            model.fit(
                max_epochs=n_init_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                delta_loss=delta_loss,
                delta_loss_batch=delta_loss_batch,
                msg=" " + str(i + 1) + "/" + str(n_init),
            )
            loss = model.get_losses()[0:n_init_epochs]

            cellstate_losses.append(loss)
            cellstate_models.append(model)

        last_delta_losses_mean = np.array(
            [
                float(torch.mean(losses[-delta_loss_batch:]))
                for losses in cellstate_losses
            ]
        )

        best_model_index = int(np.argmin(last_delta_losses_mean))
        self._state_ast = cellstate_models[best_model_index]

        self._state_ast.fit(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            delta_loss=delta_loss,
            delta_loss_batch=delta_loss_batch,
            msg=" (final)",
        )

        # Warns the user if the model has not converged
        if not self._state_ast.is_converged():
            msg = (
                "Maximum epochs reached. More iteration may be needed to"
                + " complete the training."
            )
            warnings.warn(msg)

        self._state_run_info = {
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "delta_loss": delta_loss,
            "n_init": n_init,
            "n_init_epochs": n_init_epochs,
            "delta_loss_batch": delta_loss_batch,
            "const": const,
            "dropout_rate": dropout_rate,
            "batch_norm": batch_norm,
        }

    def save_models(self, hdf5_name: str = "astir_summary.hdf5") -> None:
        """Save the summary of this model to an `hdf5` file.

        :param hdf5_name: name of the output `hdf5` file, default to "astir_summary.hdf5"
        :raises Exception: raised when this function is called before the model is trained.
        """
        if self._type_ast is None and self._state_ast is None:
            raise Exception("No model has been trained")
        with h5py.File(hdf5_name, "w") as f:
            if self._type_ast is not None:
                type_grp = f.create_group("celltype_model")

                # Storing losses
                loss_grp = type_grp.create_group("losses")
                loss_grp["losses"] = self.get_type_losses().cpu().numpy()

                # Storing parameters
                param_grp = type_grp.create_group("parameters")
                dic = list(self._type_ast.get_variables().items()) + list(
                    self._type_ast.get_data().items()
                )
                for key, val in dic:
                    param_grp[key] = val.detach().cpu().numpy()

                # Storing Recognition Network
                recog_grp = type_grp.create_group("recog_net")
                for name, param in self._type_ast.get_recognet().named_parameters():
                    if param.requires_grad:
                        recog_grp[name] = param.detach().cpu().numpy()

                # Storing fit_type argument information
                info_grp = type_grp.create_group("run_info")
                for key, val in self._type_run_info.items():
                    info_grp[key] = val

            if self._state_ast is not None:
                state_grp = f.create_group("cellstate_model")

                # Storing losses
                loss_grp = state_grp.create_group("losses")
                loss_grp["losses"] = self.get_state_losses()

                # Storing parameters
                param_grp = state_grp.create_group("parameters")
                dic = list(self._state_ast.get_variables().items()) + list(
                    self._state_ast.get_data().items()
                )

                # Storing Recognition Network
                recog_grp = state_grp.create_group("recog_net")
                for name, param in self._state_ast.get_recognet().named_parameters():
                    if param.requires_grad:
                        recog_grp[name] = param.detach().cpu().numpy()

                for key, val in dic:
                    param_grp[key] = val.detach().cpu().numpy()

                # Storing fit_state method arguments
                info_grp = state_grp.create_group("run_info")
                for key, val in self._state_run_info.items():
                    info_grp[key] = val

        if self._type_ast is not None:
            self._type_ast.get_assignment().to_hdf(
                hdf5_name, "/celltype_model/celltype_assignments"
            )
        if self._state_ast is not None:
            self._state_ast.get_assignment().to_hdf(
                hdf5_name, "/cellstate_model/cellstate_assignments"
            )

    def load_model(self, hdf5_name: str) -> None:
        """Load model from hdf5 file

        :param hdf5_name: the full path to file
        """
        has_type = False
        has_state = False
        with h5py.File(hdf5_name, "r") as f:
            if "celltype_model" in f.keys():
                run_info = f["celltype_model/run_info"]
                self._type_run_info = {
                    "learning_rate": float(np.array(run_info["learning_rate"])),
                    "n_init_epochs": int(np.array(run_info["n_init_epochs"])),
                    "batch_size": float(np.array(run_info["batch_size"])),
                    "n_init": int(np.array(run_info["n_init"])),
                    "delta_loss": float(np.array(run_info["delta_loss"])),
                    "max_epochs": int(np.array(run_info["max_epochs"])),
                }
                has_type = True
            if "cellstate_model" in f.keys():
                run_info = f["cellstate_model/run_info"]
                const = int(np.array(run_info["const"]))
                dropout_rate = float(np.array(run_info["dropout_rate"]))
                batch_norm = bool(np.array(run_info["batch_norm"]))
                self._state_run_info = {
                    "learning_rate": float(np.array(run_info["learning_rate"])),
                    "n_init_epochs": int(np.array(run_info["n_init_epochs"])),
                    "batch_size": float(np.array(run_info["batch_size"])),
                    "n_init": int(np.array(run_info["n_init"])),
                    "delta_loss": float(np.array(run_info["delta_loss"])),
                    "max_epochs": int(np.array(run_info["max_epochs"])),
                    "delta_loss_batch": int(np.array(run_info["delta_loss_batch"])),
                    "const": const,
                    "dropout_rate": dropout_rate,
                    "batch_norm": batch_norm,
                }
                has_state = True
        if has_type:
            self._type_ast = CellTypeModel(
                dset=self._type_dset,
                dtype=self._dtype,
                random_seed=np.random.randint(9999),
                device=self._device,
            )
            self._type_ast.load_hdf5(hdf5_name)
        if has_state:
            self._state_ast = CellStateModel(
                dset=self._type_dset,
                const=const,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                random_seed=np.random.randint(9999),
                dtype=self._dtype,
                device=self._device,
            )
            self._state_ast.load_hdf5(hdf5_name)

    def get_type_dataset(self) -> SCDataset:
        """Get the `SCDataset` for cell type training.

        :return: `self._type_dset`
        """
        if self._type_dset is None:
            raise Exception("the type dataset is not provided")
        return self._type_dset

    def get_state_dataset(self) -> SCDataset:
        """Get the `SCDataset` for cell state training.

        :return: `self._state_dset`
        """
        if self._state_dset is None:
            raise Exception("the state dataset is not provided")
        return self._state_dset

    def get_type_model(self) -> CellTypeModel:
        """Get the trained `CellTypeModel`.

        :raises Exception: raised when this function is called before the model is trained.
        :return: `self._type_ast`
        """
        if self._type_ast is None:
            raise Exception("The type model has not been trained yet")
        return self._type_ast

    def get_state_model(self) -> CellStateModel:
        """Get the trained `CellStateModel`.

        :raises Exception: raised when this function is celled before the model is trained.
        :return: `self._state_ast`
        """
        if self._state_ast is None:
            raise Exception("The state model has not been trained yet")
        return self._state_ast

    def get_type_run_info(self) -> Dict[str, Union[int, float]]:
        """Get the run information (i.e. `max_epochs`, `learning_rate`,
            `batch_size`, `delta_loss`, `n_init`, `n_init_epochs`) of the cell type training.

        :raises Exception: raised when this function is celled before the model is trained.
        :return: `self._type_run_info`
        """
        if self._type_run_info == {}:
            raise Exception("The type model has not been trained yet")
        return self._type_run_info

    def get_state_run_info(self) -> Dict[str, Union[int, float]]:
        """Get the run information (i.e. `max_epochs`, `learning_rate`, `batch_size`,
            `delta_loss`, `n_init`, `n_init_epochs`, `delta_loss_batch`) of the cell state training.

        :raises Exception: raised when this function is celled before the model is trained.
        :return: `self._state_run_info`
        """
        if self._state_run_info == {}:
            raise Exception("The state model has not been trained yet")
        return self._state_run_info

    def get_celltype_probabilities(self) -> pd.DataFrame:
        """Get the cell assignment probability.

        :return: `self.assignments`
        :rtype: pd.DataFrame
        """
        if self._type_ast is None:
            raise Exception("The type model has not been trained yet")
        return self._type_ast.get_assignment()

    def get_cellstates(self) -> pd.DataFrame:
        """Get cell state activations. It returns the rescaled activations,
        values between 0 and 1

        :return: state assignments
        :rtype: pd.DataFrame
        """
        if self._state_ast is None:
            raise Exception("The state model has not been trained yet")

        assign = self._state_ast.get_assignment()
        assign_min = assign.min(axis=0)
        assign_max = assign.max(axis=0)

        assign_rescale = (assign - assign_min) / (assign_max - assign_min)

        return assign_rescale

    def get_celltypes(
        self, threshold: float = 0.7, assignment_type: str = "threshold"
    ) -> pd.DataFrame:
        """Get the most likely cell type

        A cell is assigned to a cell type if the probability is greater than threshold.
        If no cell types have a probability higher than threshold, then "Unknown" is returned

        :param threshold: the probability threshold above which a cell is assigned to a cell type
        :param assignment_type: See
            :meth:`astir.CellTypeModel.get_celltypes` for full documentation
        :return: a datasets frame with most likely cell types for each
        """
        if self._type_ast is None:
            raise Exception("The type model has not been trained yet")
        return self._type_ast.get_celltypes(threshold, assignment_type=assignment_type)

    def predict_celltypes(self, dset: pd.DataFrame = None) -> pd.DataFrame:
        """Predict the probabilities of different cell type assignments.

        :param dset: the single cell protein expression dataset to predict, defaults to None
        :type dset: pd.DataFrame, optional
        :raises Exception: when the type model is not trained when this function is called
        :return: the probabilities of different cell type assignments
        :rtype: pd.DataFrame
        """
        if self._type_ast is None:
            raise Exception("The type model has not been trained yet")
        if not self._type_ast.is_converged():
            msg = "The state model has not been trained for enough epochs yet"
            warnings.warn(msg)
        if dset is None:
            dset = self.get_type_dataset()

        type_assignments = self._type_ast.predict(dset)
        type_assignments.columns = dset.get_classes() + ["Other"]
        type_assignments.index = dset.get_cell_names()
        return type_assignments

    def predict_cellstates(self, dset: SCDataset = None) -> pd.DataFrame:
        """Get the prediction cell state activations on a dataset on an
        existing model

        :param new_dset: the dataset to predict cell state activations, default to None

        :return: the prediction of cell state activations
        """
        if self._state_ast is None:
            raise Exception("The state model has not been trained yet")

        if not self._state_ast.is_converged():
            msg = "The state model has not been trained for enough epochs yet"
            warnings.warn(msg)

        if self._state_dset is None:
            raise NotClassifiableError(
                "Dataset or marker for cell state classification is not provided"
            )

        g = self._state_ast.get_final_mu_z(dset).detach().cpu().numpy()

        state_assignments = pd.DataFrame(g)

        assign_min = state_assignments.min(axis=0)
        assign_max = state_assignments.max(axis=0)

        assign_rescale = (state_assignments - assign_min) / (assign_max - assign_min)

        assign_rescale.columns = self._state_dset.get_classes()
        assign_rescale.index = self._state_dset.get_cell_names()

        return assign_rescale

    def assign_celltype_hierarchy(self, depth: int = 1) -> pd.DataFrame:
        """Get cell type assignment at a specified higher hierarchy according to the hierarchy provided
            in the dictionary.

        :param depth: the depth of hierarchy to assign probability to, defaults to 1
        :type depth: int, optional
        :raises Exception: raised when the dictionary for hierarchical structure is not provided
            or the model hasn't been trained.
        :return: probability assignment of cell type at a superstructure
        :rtype: pd.DataFrame
        """
        if self._type_ast is None:
            raise Exception("The type model has not been trained yet")
        if self._hierarchy_dict is None:
            raise Exception("The dictionary for hierarchical structure is not provided")
        prob = self.get_celltype_probabilities()
        hier_df = pd.DataFrame()
        self._assign_celltype_hierarchy_helper1(
            hier_df, prob, self._hierarchy_dict, depth
        )
        return hier_df

    def _assign_celltype_hierarchy_helper1(
        self,
        hier_df: Union[pd.DataFrame, list],
        prob: pd.DataFrame,
        dic: dict,
        depth: int = 1,
    ) -> None:
        """Helper for `assign_celltype_hierarchy`, calculates summed probabilities recursively."""
        if depth == 1:
            for key, val in dic.items():
                hier_df[key] = self._assign_celltype_hierarchy_helper2(prob, val)
        else:
            for key, cells in dic.items():
                if isinstance(cells, list):
                    for cell in cells:
                        hier_df[cell] = prob[cell]
                else:
                    self._assign_celltype_hierarchy_helper1(
                        hier_df, prob, cells, depth - 1
                    )

    def _assign_celltype_hierarchy_helper2(
        self, prob: pd.DataFrame, dic: dict
    ) -> pd.Series:
        """Helper for `assign_celltype_hierarchy`, calculates summed probability for
        cells under the given dict.
        """
        if isinstance(dic, list):
            return prob[dic].sum(axis=1)
        temp_df = pd.DataFrame()
        for key, cells in dic.items():
            temp_df[key] = self._assign_celltype_hierarchy_helper2(prob, cells)
        return temp_df.sum(axis=1)

    def type_clustermap(
        self,
        plot_name: str = "celltype_protein_cluster.png",
        threshold: float = 0.7,
        figsize: Tuple[float, float] = (7, 5),
        prob_assign: Optional[pd.DataFrame] = None,
    ) -> None:
        """Save the heatmap of protein content in cells with cell types labeled.

        :param plot_name: name of the plot, extension(e.g. .png or .jpg) is needed, defaults to "celltype_protein_cluster.png"
        :type plot_name: str, optional
        :param threshold: the probability threshold above which a cell is assigned to a cell type, defaults to 0.7
        :type threshold: float, optional
        """
        if self._type_ast is None:
            raise Exception("The type model has not been trained yet")
        self._type_ast.plot_clustermap(plot_name, threshold, figsize, prob_assign)

    def get_hierarchy_dict(self) -> Dict[str, List[str]]:
        """Get the dictionary for cell type hierarchical structure.

        :return: `self._hierarchy_dict`
        :rtype: Dict[str, List[str]]
        """
        if self._hierarchy_dict is None:
            raise Exception("the hierarchical marker is not provided")
        return self._hierarchy_dict

    def get_type_losses(self) -> np.array:
        """Get the final losses of the type model.

        :return: `self.losses`
        :rtype: np.array
        """
        if self._type_ast is None:
            raise Exception("The type model has not been trained yet")
        return self._type_ast.get_losses()

    def get_state_losses(self) -> np.array:
        """Getter for losses

        :return: a numpy array of losses for each training iteration the\
            model runs
        :rtype: np.array
        """
        if self._state_ast is None:
            raise Exception("The state model has not been trained yet")
        return self._state_ast.get_losses()

    def type_to_csv(
        self,
        output_csv: str,
        threshold: float = 0.7,
        assignment_type: str = "threshold",
    ) -> None:
        """Save the cell type assignemnt to a `csv` file.

        :param output_csv: name for the output .csv file
        :param assignment_type: See
            :meth:`astir.CellTypeModel.get_celltypes` for full documentation
        :type output_csv: str
        """
        self.get_celltypes(threshold, assignment_type=assignment_type).to_csv(
            output_csv
        )

    def state_to_csv(self, output_csv: str) -> None:
        """Writes state assignment output from training state model in csv
        file

        :param output_csv: path to output csv
        :type output_csv: str, required
        """
        self.get_cellstates().to_csv(output_csv)

    def __str__(self) -> str:
        """String representation of Astir object"""
        msg = "Astir object"
        l = 0
        if self._type_dset is not None:
            msg += ", " + str(self._type_dset.get_n_classes()) + " cell types"
            l = len(self._type_dset)
        if self._state_dset is not None:
            msg += ", " + str(self._state_dset.get_n_classes()) + " cell states"
            l = len(self._state_dset)
        if l != 0:
            msg += ", " + str(l) + " cells"
        return msg

    def diagnostics_celltype(
        self, threshold: float = 0.5, alpha: float = 0.01
    ) -> pd.DataFrame:
        """Run diagnostics on cell type assignments

        This performs a basic test that cell types express their markers at
        higher levels than in other cell types. This function performs the following steps:

        1. Iterates through every cell type and every marker for that cell type
        2. Given a cell type *c* and marker *g*, find the set of cell
            types *D* that don't have *g* as a marker
        3. For each cell type *d* in *D*, perform a t-test between the
            expression of marker *g* in *c* vs *d*
        4. If *g* is not expressed significantly higher (at significance
            *alpha*), output a diagnostic explaining this for further investigation.

        :param threshold: The threshold at which cell types are assigned (see `get_celltypes`)
        :param alpha: The significance threshold for t-tests for determining over-expression
        :return: Either a :class:`pd.DataFrame` listing cell types whose markers
            aren't expressed signficantly higher.
        """
        celltypes = list(self.get_celltypes(threshold=threshold)["cell_type"])
        return self.get_type_model().diagnostics(celltypes, alpha=alpha)

    def diagnostics_cellstate(self) -> pd.DataFrame:
        """Run diagnostics on cell state assignments

        This performs a basic test by comparing the correlation values
        between all marker genes and all non marker genes. It detects where the
        non marker gene has higher correlation values than the smallest
        correlation values of marker genes.

        1. Get correlations between all cell states and proteins
        2. For each cell state *c*, get the smallest correlation with marker *g*
        3. For each cell state *c* and its non marker *g*, find any correlation that is
            bigger than those smallest correlation for *c*.
        4. Any *c* and *g* pairs found in step 3 will be included in the output of
            `Astir.diagnostics_cellstate()`, including an explanation.

        :return: diagnostics
        """
        return self.get_state_model().diagnostics()

    def normalize(self, percentile_lower: int = 1, percentile_upper: int = 99) -> None:
        """Normalize the expression datasets

        This performs a two-step normalization:
        1. A `log(1+x)` transformation to the datasets
        2. Winsorizes to (`percentile_lower`, `percentile_upper`)

        :param percentile_lower: Lower percentile for winsorization
        :param percentile_upper: Upper percentile for winsorization
        """
        self.get_type_dataset().normalize(percentile_lower, percentile_upper)
        self.get_state_dataset().normalize(percentile_lower, percentile_upper)


class NotClassifiableError(RuntimeError):
    """Raised when the input datasets is not classifiable."""

    pass


# KC removed
# :param assignments: cell type assignment probabilities
# :param losses: losses after optimization
# :param type_dict: dictionary mapping cell type
#     to the corresponding genes
# :param state_dict: dictionary mapping cell state
#     to the corresponding genes
# :param cell_types: list of all cell types from marker
# :param mtype_genes: list of all cell type genes from marker
# :param mstate_genes: list of all cell state genes from marker
# :param N: number of rows of datasets
# :param G: number of cell type genes
# :param C: number of cell types
# :param initializations: initialization parameters
# :param datasets: parameters that is not to be optimized
# :param variables: parameters that is to be optimized
