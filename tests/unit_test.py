#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% Imports

import unittest

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from dhs.helpers import create_dhs
from dhs.utils import NumFormat, _PmObj

# %% Class


class UnitTest(unittest.TestCase):
    def test_create_dhs_returns_pymc_model(self) -> None:
        """
        Perform unit test #1.

        Check if the returned object is an instance of PyMC Model.
        """
        model = create_dhs(
            target=pd.Series([1, 2, 3]),
            features=np.array([[2, 0.5], [4, 1], [6, 1.5]]),
        )

        self.assertIsInstance(model, pm.Model)

    def test_create_normal_has_mean_zero(self) -> None:
        """
        Perform unit test #2.

        Check if the sampled standard normal prior has zero mean.
        """
        with pm.Model():
            _PmObj.create_normal(name="normal")
            trace = pm.sample(draws=2000, tune=1000, chains=2)
            summary = az.summary(data=trace, hdi_prob=0.95)
            hdi_lower, hdi_upper = summary.iloc[0, 2:4]

        self.assertGreaterEqual(hdi_upper, 0)
        self.assertLessEqual(hdi_lower, 0)

    def test_formatting_behavior(self) -> None:
        """
        Perform unit test #3.

        Check if the formatted value equals target value.
        """

        self.assertEqual("1", NumFormat.format_num(in_val=1.00))


if __name__ == "__main__":
    unittest.main()
