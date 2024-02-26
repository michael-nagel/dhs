#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

import pandas as pd

# Function


def _create_df_metrics(
    dict_var: dict, simul: int, vars_2: list
) -> pd.DataFrame:
    """
    Generate a DataFrame from dictionary containing metrics.

    Parameters
    ----------
    dict_var : dict
        Dictionary containing metrics.
    simul : int
        Simulation study (1 or 2).
    vars_2 : list
        Second variable for the simulation study (signal strength or
        feature correlation).

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics.
    """
    if simul not in [1, 2]:
        raise ValueError("simul must be either 1 or 2")

    if simul == 1:
        keys = [
            f"$A={vars_2[0]}$",
            f"$A={vars_2[0]}$",
            f"$A={vars_2[1]}$",
            f"$A={vars_2[1]}$",
            f"$A={vars_2[2]}$",
            f"$A={vars_2[2]}$",
        ]
    elif simul == 2:
        keys = [
            f"$\\rho={vars_2[0]:.2f}$",
            f"$\\rho={vars_2[0]:.2f}$",
            f"$\\rho={vars_2[1]:.2f}$",
            f"$\\rho={vars_2[1]:.2f}$",
            f"$\\rho={vars_2[2]:.2f}$",
            f"$\\rho={vars_2[2]:.2f}$",
        ]

    df_metrics = pd.concat(
        objs=[
            dict_var[list(dict_var)[0]],
            pd.DataFrame(index=[""]),
            dict_var[list(dict_var)[1]],
            pd.DataFrame(index=[""]),
            dict_var[list(dict_var)[2]],
        ],
        axis=0,
        keys=keys,
    )

    return df_metrics
