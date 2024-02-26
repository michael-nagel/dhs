#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

import arviz as az
import numpy as np

from dhs.utils import create_func_dict

# Function


def _append_metrics(
    trace: az.InferenceData,
    metrics: dict,
    var_1: int,
    var_2: int,
    prior: str,
    i: int,
    params: np.ndarray,
    n_rel_params: int,
    irrel_params_idxs: np.ndarray,
    hdi: float,
    dim: int | None = None,
    n_vars: int | None = None,
) -> None:
    """
    Append metrics to dictionary.

    Parameters
    ----------
    trace : az.InferenceData
        Record of the sampling process.
    metrics : dict
        Metrics dictionary.
    var_1 : int
        First variable for the simulation study.
    var_2 : int
        Second variable for the simulation study.
    prior : str
        Prior distribution.
    i : int
        For loop counter.
    params : np.ndarray
        Parameters
    n_rel_params : int
        Number of relevant parameters.
    irrel_params_idxs : np.ndarray
        Irrelevant parameter indices.
    hdi : float
        Highest density interval.
    dim : int, optional
        Dimensionality.
    n_vars : int, optional
        Number of variables.
    """
    # Summary stats
    summary = az.summary(
        trace,
        hdi_prob=hdi,
        stat_funcs=create_func_dict(),
        var_names=["theta"],
    )

    metrics["Loss"][var_1][var_2][prior] += [
        ((summary["median"] - params[:, i]) ** 2).sum()
    ]

    metrics["StdMed"][var_1][var_2][prior] += [summary["std_median"].mean()]

    metrics["Coverage"][var_1][var_2][prior] += [
        (
            (summary.iloc[:, 2] <= params[:, i])
            & (summary.iloc[:, 3] >= params[:, i])
        ).sum()
    ]

    metrics["Rhat"][var_1][var_2][prior] += [summary["r_hat"].mean()]

    if dim is None and n_vars is not None:
        metrics["Type1Err"][var_1][var_2][prior] += [
            (n_vars - n_rel_params)
            - (
                (summary.iloc[irrel_params_idxs, 2] <= 0)
                & (summary.iloc[irrel_params_idxs, 3] >= 0)
            ).sum()
        ]

    elif dim is not None:
        metrics["Type1Err"][var_1][var_2][prior] += [
            (dim - n_rel_params)
            - (
                (summary.iloc[irrel_params_idxs, 2] <= 0)
                & (summary.iloc[irrel_params_idxs, 3] >= 0)
            ).sum()
        ]

    stats = trace["sample_stats"]
    metrics["Minutes"][var_1][var_2][prior] += [
        trace.attrs["sampling_time"] / 60
    ]

    metrics["Diverg"][var_1][var_2][prior] += [stats.diverging.values.sum()]
