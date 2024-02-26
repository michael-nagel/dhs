#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

import pandas as pd

# Function


def _append_to_dict(
    metrics_tot: dict,
    dict_var: str,
    var_2: float,
    keys: list,
    vals: list,
    vars_2: list,
    i: int,
) -> None:
    """
    Append pandas.DataFrame to dictionary.

    Parameters
    ----------
    metrics_tot: dict
        Dictionary containing all metrics.
    dict_var : str
        Dictionary name.
    var_2 : float
        Second variable's element for the simulation study.
    keys : list
        Dictionary keys.
    vals : list
        Dictionary values.
    vars_2: list
        Second variable for the simulation study signal strength or
        feature correlation).
    i: int
        For loop counter.
    """
    metrics_tot[dict_var][var_2] = pd.DataFrame(
        {
            keys[0]: vals[0].loc[vars_2[i], :],
            keys[1]: vals[1].loc[vars_2[i], :],
            keys[2]: vals[2].loc[vars_2[i], :],
            keys[3]: vals[3].loc[vars_2[i], :],
        }
    )
