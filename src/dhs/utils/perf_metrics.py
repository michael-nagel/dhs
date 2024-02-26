# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

from typing import Tuple

import numpy as np
import pandas as pd

# Class


class PerfMetrics:
    def __init__(self) -> None:
        pass

    @staticmethod
    def calc_avg_loss(data: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate the average loss.

        Parameters
        ----------
        data : dict
            Data from which the average metric is calculated.

        Returns
        ----------
        avg_loss : pandas.DataFrame
            Average loss.
        rel_avg_std_loss : pandas.DataFrame
            Relative average standard deviation of the loss.
        """
        df = pd.DataFrame(data)
        avg_loss = df[df.columns].applymap(np.mean)
        rel_avg_std_loss = df[df.columns].applymap(np.std) / avg_loss

        return avg_loss.T, rel_avg_std_loss.T

    @staticmethod
    def calc_avg_metric(data: dict, **kwargs) -> pd.DataFrame:
        """
        Calculate the average metric of interest.

        Parameters
        ----------
        data : dict
            Data from which the average metric is calculated.
        **kwargs
            These parameters will be passed to the function.

        Returns
        ----------
        avg_metric : pandas.DataFrame
            Average metric.
        """
        df = pd.DataFrame(data)

        n_repl = kwargs.get("n_repl")
        n_vars = kwargs.get("n_vars")

        if n_vars is None:
            avg_metric = df[df.columns].applymap(lambda x: np.sum(x) / len(x))
        else:
            avg_metric = df[df.columns].applymap(
                lambda x: np.sum(x) / (n_repl * n_vars)
            )

        return avg_metric.T

    @staticmethod
    def calc_avg_std(data: dict) -> pd.DataFrame:
        """
        Calculate the average standard deviation.

        Parameters
        ----------
        data : dict
            Data from which the average metric is calculated.

        Returns
        ----------
        avg_std : pandas.DataFrame
            Average standard deviation.
        """
        df = pd.DataFrame(data)
        avg_std = df[df.columns].applymap(np.mean)

        return avg_std.T
