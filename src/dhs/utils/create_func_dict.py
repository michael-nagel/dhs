#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

import numpy as np

# Function


def create_func_dict() -> dict:
    """
    Create dictionary with summary metrics.

    This function creates a dictionary that contains the median and the
    standard deviation of the median for each column of a DataFrame.

    Returns
    -------
    dict
        Dictionary with pd.Series.
    """

    def calc_std_median(x):
        return np.sqrt(np.mean((x - np.median(x)) ** 2))

    return {
        "median": np.median,
        "std_median": calc_std_median,
    }
