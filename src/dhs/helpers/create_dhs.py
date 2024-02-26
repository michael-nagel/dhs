#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

import numpy as np
import pandas as pd
import pymc as pm

from dhs.utils import _PmObj

# Model


def create_dhs(
    target: pd.Series | np.ndarray, features: np.ndarray
) -> pm.Model:
    """
    Create Dirichlet-horseshoe prior.

    This function creates a general model for estimation with the
    Dirichlet-horseshoe prior. Modifications can be made through
    function inputs, or by adjusting the hyperparameters, e.g., the
    slab width, of the specified priors. Experienced users may go
    beyond adjusting hyperparameters. For instance, they can introduce
    additional priors to accommodate random effects or alter the
    distribution from which the noise parameter is sampled. It is also
    possible to fix values instead of sampling them.

    Parameters
    ----------
    target : pd.Series | np.ndarray
        Dependent variable.
    features : np.ndarray
        Predictor variables.

    Returns
    -------
    pm.Model
        Dirichlet-horseshoe PyMC model to be estimated.
    """
    assert isinstance(
        target, (pd.Series, np.ndarray)
    ), "target must be a pd.Series or np.ndarray"
    assert isinstance(features, np.ndarray), "features must be a np.ndarray"

    n_obs, dim = features.shape

    with pm.Model() as model:
        # Model error
        noise = _PmObj.create_halfcauchy(name="sig_eps")

        # Intercept (optional)
        intc = _PmObj.create_normal(name="alpha")

        # Sparsity parameter for both concentration rate and fraction of
        # relevant variables
        spars = _PmObj.create_beta(name="eta")

        # Number of relevant variables
        n_rel_vars = dim * spars

        # Joint regularization prior
        joint_reg_ofst = _PmObj.create_dirichlet(
            name="phi", comp=dim, concen=spars
        )
        joint_reg = _PmObj.create_det(name="phi_tld", var=dim * joint_reg_ofst)

        # Global regularization prior
        glob_reg_scale = (n_rel_vars / (dim - n_rel_vars)) * (
            noise / np.sqrt(n_obs)
        )
        glob_reg = _PmObj.create_halfcauchy(name="tau", beta=glob_reg_scale)

        # Slab width
        slab = _PmObj.create_invgamma(name="c_sq")

        # Local regularization prior
        loc_reg_ofst = _PmObj.create_halfcauchy(name="lam", shape=dim)
        loc_reg = _PmObj.create_det(
            name="lam_tld",
            var=(
                slab
                * pm.math.sqr(loc_reg_ofst)
                / (slab + pm.math.sqr(glob_reg) * pm.math.sqr(loc_reg_ofst))
            ),
        )

        # Regression coefficients (uncentered representation)
        param_ofst = _PmObj.create_normal(name="beta_ofst", shape=dim)
        param = _PmObj.create_det(
            name="beta",
            var=param_ofst * joint_reg * pm.math.sqrt(loc_reg) * glob_reg,
        )

        # Target prediction
        target_pred = _PmObj.create_det(
            name="y_hat", var=pm.math.dot(features, param) + intc
        )

        # Likelihood
        _PmObj.create_normal(
            name="likelihood", mu=target_pred, sigma=noise, observed=target
        )

    return model
