#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

import numpy as np
import pymc as pm

from dhs.utils import _PmObj

# Function


def create_mdl_alc_data(dict_data: dict) -> pm.Model:
    """
    Create a PyMC model using the Dirichlet-horseshoe prior.

    This function creates a random intercept model that uses the
    Dirichlet-horseshoe prior for estimation.

    Parameters
    ----------
    dict_data : dict
        Data dictionary.

    Returns
    -------
    pm.Model
        Real-data PyMC model to be estimated.
    """
    with pm.Model(coords=dict_data["coords"]) as model:
        # Matchup indices
        matchup_idx = pm.Data(
            "matchup_idx",
            dict_data["matchup_idxs"],
            dims="obs_id",
            mutable=True,
        )

        # Model error
        sig_eps = _PmObj.create_halfcauchy(name="sig_eps")

        # Random intercept
        mean_rnd_icpt = _PmObj.create_normal(name="v_2")
        sig_rnd_icpt = _PmObj.create_halfnormal(name="sig_u_2")
        rnd_icpt_ofst = _PmObj.create_normal(
            name="u_2", shape=dict_data["n_matches"]
        )
        rnd_icpt = mean_rnd_icpt + rnd_icpt_ofst * sig_rnd_icpt

        # Sparsity parameter for both concentration rate and fraction of
        #  relevant variables
        spars_covars_2 = _PmObj.create_beta(name="nu_gamma_2")
        spars_covars_1 = _PmObj.create_beta(name="nu_gamma_1")
        spars_sur = _PmObj.create_beta(name="nu_beta_11")
        spars_sus = _PmObj.create_beta(name="nu_beta_12")
        spars_time_dum = _PmObj.create_beta(name="nu_psi_1")
        spars_lag_target = _PmObj.create_beta(name="nu_phi_1")

        # Total number of relevant variables
        n_rel_vars = (
            spars_covars_2 * dict_data["n_covars_2"]
            + spars_covars_1 * dict_data["n_covars_1"] * dict_data["n_states"]
            + spars_sur * dict_data["n_sur_vars"] * dict_data["n_states"]
            + spars_sus * dict_data["n_sus_vars"] * dict_data["n_states"]
            + spars_time_dum * dict_data["n_time_dum"] * dict_data["n_states"]
            + spars_lag_target
            * dict_data["n_lag_target_vars"]
            * dict_data["n_states"]
        )

        # Joint regularization priors
        joint_reg_covars_2_ofst = _PmObj.create_dirichlet(
            name="kappa_gamma_2_tld",
            comp=dict_data["n_covars_2"],
            concen=spars_covars_2,
        )

        joint_reg_covars_2 = _PmObj.create_det(
            name="kappa_gamma_2",
            var=joint_reg_covars_2_ofst * dict_data["n_covars_2"],
        )

        joint_reg_covars_1_ofst = _PmObj.create_dirichlet(
            name="kappa_gamma_1_tdl",
            comp=dict_data["n_covars_1"],
            concen=spars_covars_1,
            shape=(dict_data["n_states"], dict_data["n_covars_1"]),
        )

        joint_reg_covars_1 = _PmObj.create_det(
            name="kappa_gamma_1",
            var=joint_reg_covars_1_ofst
            * dict_data["n_states"]
            * dict_data["n_covars_1"],
        )

        joint_reg_sur_ofst = _PmObj.create_dirichlet(
            name="kappa_beta_11_tld",
            comp=dict_data["n_sur_vars"],
            concen=spars_sur,
            shape=(dict_data["n_states"], dict_data["n_sur_vars"]),
        )

        joint_reg_sur = _PmObj.create_det(
            name="kappa_beta_11",
            var=joint_reg_sur_ofst
            * dict_data["n_states"]
            * dict_data["n_sur_vars"],
        )

        joint_reg_sus_ofst = _PmObj.create_dirichlet(
            name="kappa_beta_12_tld",
            comp=dict_data["n_sus_vars"],
            concen=spars_sus,
            shape=(dict_data["n_states"], dict_data["n_sus_vars"]),
        )

        joint_reg_sus = _PmObj.create_det(
            name="kappa_beta_12",
            var=joint_reg_sus_ofst
            * dict_data["n_states"]
            * dict_data["n_sus_vars"],
        )

        joint_reg_time_dum_ofst = _PmObj.create_dirichlet(
            name="kappa_psi_1_tld",
            comp=dict_data["n_time_dum"],
            concen=spars_time_dum,
            shape=(dict_data["n_states"], dict_data["n_time_dum"]),
        )

        joint_reg_time_dum = _PmObj.create_det(
            name="kappa_psi_1",
            var=joint_reg_time_dum_ofst
            * dict_data["n_states"]
            * dict_data["n_time_dum"],
        )

        joint_reg_lag_target_ofst = _PmObj.create_dirichlet(
            name="kappa_phi_1_tld",
            comp=dict_data["n_lag_target_vars"],
            concen=spars_lag_target,
            shape=(dict_data["n_states"], dict_data["n_lag_target_vars"]),
        )

        joint_reg_lag_target = _PmObj.create_det(
            name="kappa_phi_1",
            var=joint_reg_lag_target_ofst
            * dict_data["n_states"]
            * dict_data["n_lag_target_vars"],
        )

        # Global regularization priors
        n_vars_state = (
            dict_data["n_covars_1"]
            + dict_data["n_sur_vars"]
            + dict_data["n_sus_vars"]
            + dict_data["n_time_dum"]
            + dict_data["n_lag_target_vars"]
        ) * dict_data["n_states"] + dict_data["n_covars_2"]

        glob_reg_scale = (n_rel_vars / (n_vars_state - n_rel_vars)) * (
            sig_eps / np.sqrt(dict_data["target"].shape[0])
        )
        glob_reg = _PmObj.create_halfcauchy(name="tau", beta=glob_reg_scale)
        slab_var = _PmObj.create_invgamma(name="c_sq", alpha=3, beta=1)

        # Local regularization priors
        loc_reg_covars_2_ofst = _PmObj.create_halfcauchy(
            name="lambda_gamma_2_tld", shape=dict_data["n_covars_2"]
        )
        loc_reg_covars_2 = _PmObj.create_det(
            name="lambda_gamma_2",
            var=(
                slab_var
                * pm.math.sqr(loc_reg_covars_2_ofst)
                / (
                    slab_var
                    + pm.math.sqr(glob_reg)
                    * pm.math.sqr(loc_reg_covars_2_ofst)
                )
            ),
        )

        loc_reg_covars_1_ofst = _PmObj.create_halfcauchy(
            name="lambda_gamma_1_tld",
            shape=(dict_data["n_states"], dict_data["n_covars_1"]),
        )

        loc_reg_covars_1 = _PmObj.create_det(
            name="lambda_gamma_1",
            var=(
                slab_var
                * pm.math.sqr(loc_reg_covars_1_ofst)
                / (
                    slab_var
                    + pm.math.sqr(glob_reg)
                    * pm.math.sqr(loc_reg_covars_1_ofst)
                )
            ),
        )

        loc_reg_sur_ofst = _PmObj.create_halfcauchy(
            name="lambda_beta_11_tld",
            shape=(dict_data["n_states"], dict_data["n_sur_vars"]),
        )

        loc_reg_sur = _PmObj.create_det(
            name="lambda_beta_11",
            var=(
                slab_var
                * pm.math.sqr(loc_reg_sur_ofst)
                / (
                    slab_var
                    + pm.math.sqr(glob_reg) * pm.math.sqr(loc_reg_sur_ofst)
                )
            ),
        )

        loc_reg_sus_ofst = _PmObj.create_halfcauchy(
            name="lambda_beta_12_tld",
            shape=(dict_data["n_states"], dict_data["n_sus_vars"]),
        )

        loc_reg_sus = _PmObj.create_det(
            name="lambda_beta_12",
            var=(
                slab_var
                * pm.math.sqr(loc_reg_sus_ofst)
                / (
                    slab_var
                    + pm.math.sqr(glob_reg) * pm.math.sqr(loc_reg_sus_ofst)
                )
            ),
        )

        loc_reg_time_dum_ofst = _PmObj.create_halfcauchy(
            name="lambda_psi_1_tld",
            shape=(dict_data["n_states"], dict_data["n_time_dum"]),
        )

        loc_reg_time_dum = _PmObj.create_det(
            name="lambda_psi_1",
            var=(
                slab_var
                * pm.math.sqr(loc_reg_time_dum_ofst)
                / (
                    slab_var
                    + pm.math.sqr(glob_reg)
                    * pm.math.sqr(loc_reg_time_dum_ofst)
                )
            ),
        )

        loc_reg_lag_target_ofst = _PmObj.create_halfcauchy(
            name="lambda_phi_1_tld",
            shape=(dict_data["n_states"], dict_data["n_lag_target_vars"]),
        )

        loc_reg_lag_target = _PmObj.create_det(
            name="lambda_phi_1",
            var=(
                slab_var
                * pm.math.sqr(loc_reg_lag_target_ofst)
                / (
                    slab_var
                    + pm.math.sqr(glob_reg)
                    * pm.math.sqr(loc_reg_lag_target_ofst)
                )
            ),
        )

        # Level 2 predictors
        param_covars_2_ofst = _PmObj.create_normal(
            name="gamma_2_tld", shape=dict_data["n_covars_2"]
        )
        param_covars_2 = _PmObj.create_det(
            name="gamma_2",
            var=param_covars_2_ofst
            * joint_reg_covars_2
            * pm.math.sqrt(loc_reg_covars_2)
            * glob_reg,
        )

        # Level 1 intercepts
        icpt_1 = _PmObj.create_det(
            name="v_1",
            var=pm.math.dot(dict_data["covars_2"], param_covars_2) + rnd_icpt,
        )

        # Surprise
        param_sur_ofst = _PmObj.create_normal(
            name="beta_11_tld",
            shape=(dict_data["n_states"], dict_data["n_sur_vars"]),
        )

        param_sur = _PmObj.create_det(
            name="beta_11",
            var=param_sur_ofst
            * joint_reg_sur
            * pm.math.sqrt(loc_reg_sur)
            * glob_reg,
        )

        param_sur_state = param_sur[
            np.tile(
                dict_data["states"],
                int(dict_data["n_sur_vars"] / (dict_data["n_lags"] + 1)),
            ),
            np.arange(0, dict_data["n_sur_vars"]),
        ]

        # Suspense
        param_sus_ofst = _PmObj.create_normal(
            name="beta_12_tld",
            shape=(dict_data["n_states"], dict_data["n_sus_vars"]),
        )

        param_sus = _PmObj.create_det(
            name="beta_12",
            var=param_sus_ofst
            * joint_reg_sus
            * pm.math.sqrt(loc_reg_sus)
            * glob_reg,
        )

        param_sus_state = param_sus[
            np.tile(
                dict_data["states"],
                int(dict_data["n_sus_vars"] / (dict_data["n_lags"] + 1)),
            ),
            np.arange(0, dict_data["n_sus_vars"]),
        ]

        # Level 1 predictors
        param_covars_1_ofst = _PmObj.create_normal(
            name="gamma_1_tld",
            shape=(dict_data["n_states"], dict_data["n_covars_1"]),
        )

        param_covars_1 = _PmObj.create_det(
            name="gamma_1",
            var=param_covars_1_ofst
            * joint_reg_covars_1
            * pm.math.sqrt(loc_reg_covars_1)
            * glob_reg,
        )

        param_covars_1_state = param_covars_1[
            np.tile(
                dict_data["states"],
                int(dict_data["n_covars_1"] / (dict_data["n_lags"] + 1)),
            ),
            np.arange(0, dict_data["n_covars_1"]),
        ]

        # Time dummies
        param_time_dum_ofst = _PmObj.create_normal(
            name="psi_1_tld",
            shape=(dict_data["n_states"], dict_data["n_time_dum"]),
        )

        param_time_dum = _PmObj.create_det(
            name="psi_1",
            var=param_time_dum_ofst
            * joint_reg_time_dum
            * pm.math.sqrt(loc_reg_time_dum)
            * glob_reg,
        )

        param_time_dum_state = param_time_dum[dict_data["states"][:, 0]]

        # Lagged dependent variables
        param_lag_target_ofst = _PmObj.create_normal(
            name="phi_1_tld",
            shape=(dict_data["n_states"], dict_data["n_lag_target_vars"]),
        )

        param_lag_target = _PmObj.create_det(
            name="phi_1",
            var=param_lag_target_ofst
            * joint_reg_lag_target
            * pm.math.sqrt(loc_reg_lag_target)
            * glob_reg,
        )

        param_lag_target_state = param_lag_target[
            dict_data["states"][:, 1:],
            np.arange(0, dict_data["n_lag_target_vars"]),
        ]

        # Model mean
        target_pred = (
            pm.math.sum(dict_data["sur"] * param_sur_state, axis=1)
            + pm.math.sum(dict_data["sus"] * param_sus_state, axis=1)
            + pm.math.sum(
                dict_data["lag_target"] * param_lag_target_state, axis=1
            )
            + pm.math.sum(dict_data["covars_1"] * param_covars_1_state, axis=1)
            + pm.math.sum(dict_data["time_dum"] * param_time_dum_state, axis=1)
            + icpt_1[matchup_idx]
        )

        # Model likelihood
        _PmObj.create_normal(
            name="y_like",
            mu=target_pred,
            sigma=sig_eps,
            observed=dict_data["target"],
        )

        return model
