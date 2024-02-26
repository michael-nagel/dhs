# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

from collections import defaultdict
from typing import Any, DefaultDict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import lognorm

from dhs.helpers.base import PriorMdls, _append_metrics
from dhs.utils import (
    PerfMetrics,
    _append_to_dict,
    _create_df_metrics,
    run_sampler,
)

# Function


def create_simuls(
    simul: int,
    vars_1: list,
    vars_2: list,
    priors: list,
    n_repl: int,
    seed: int,
    n_draws: int,
    n_tune: int,
    n_chains: int,
    targ_acpt: float,
    hdi: float,
    dim: int | None = None,
    n_vars: int | None = None,
    frac_rel_vars: float | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run simulation studies.

    This function performs the simulation studies 1 and 2.

    Parameters
    ----------
    simul : int
        Simulation study (1 or 2).
    vars_1 : list
        First variable for the simulation study.
    vars_2 : list
        Second variable for the simulation study.
    priors : list
        Prior distributions.
    n_repl : int
        Number of replications.
    seed : int
        Random seed.
    n_draws : int
        Number of draws.
    n_tune : int
        Number of tuning samples (burn-in).
    n_chains : int
        Number of chains.
    targ_acpt: float
        Target acceptance rate.
    hdi: float
        Highest density interval
    dim : int, optional
        Dimensionality.
    n_vars : int, optional
        Number of variables.
    frac_rel_vars : float, optional
        Fraction of relevant variables.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    metric_names = [
        "Loss",
        "StdMed",
        "Coverage",
        "Type1Err",
        "Rhat",
        "Minutes",
        "Diverg",
    ]

    metrics: DefaultDict[Any, Any] = defaultdict()

    for metric_name in metric_names:
        metrics[metric_name] = {
            var_1: {var_2: {prior: [] for prior in priors} for var_2 in vars_2}
            for var_1 in vars_1
        }

    if simul == 1 and dim is not None:
        # Error term
        white_noise = np.random.normal(loc=0, scale=1, size=[dim, n_repl])

        for var_1 in vars_1:
            for var_2 in vars_2:
                # Generate parameters and randomly set a some to zero
                params = np.ones(shape=[dim, n_repl]) * var_2
                n_rel_params = int(dim * var_1 / 100)
                irrel_params_idxs = np.random.choice(
                    dim, dim - n_rel_params, replace=False
                )
                params[irrel_params_idxs, :] = 0
                # Calculate the target
                target = params + white_noise
                for prior in priors:
                    for i in range(0, n_repl):
                        prior_mdls = PriorMdls(target=target[:, i])
                        if prior == "BAL":
                            trace = run_sampler(
                                model=prior_mdls.create_mdl_bal(),
                                n_draws=n_draws,
                                n_tune=n_tune,
                                n_chains=n_chains,
                                seed=seed,
                                targ_acpt=targ_acpt,
                            )
                        elif prior == "DHS":
                            trace = run_sampler(
                                model=prior_mdls.create_mdl_dhs(),
                                n_draws=n_draws,
                                n_tune=n_tune,
                                n_chains=n_chains,
                                seed=seed,
                                targ_acpt=targ_acpt,
                            )
                        elif prior == "DL":
                            trace = run_sampler(
                                model=prior_mdls.create_mdl_dl(),
                                n_draws=n_draws,
                                n_tune=n_tune,
                                n_chains=n_chains,
                                seed=seed,
                                targ_acpt=targ_acpt,
                            )
                        elif prior == "HS":
                            trace = run_sampler(
                                model=prior_mdls.create_mdl_hs(),
                                n_draws=n_draws,
                                n_tune=n_tune,
                                n_chains=n_chains,
                                seed=seed,
                                targ_acpt=targ_acpt,
                            )
                        elif prior == "RHS":
                            trace = run_sampler(
                                model=prior_mdls.create_mdl_rhs(),
                                n_draws=n_draws,
                                n_tune=n_tune,
                                n_chains=n_chains,
                                seed=seed,
                                targ_acpt=targ_acpt,
                            )
                        else:
                            trace = run_sampler(
                                model=prior_mdls.create_mdl_spsl(),
                                n_draws=n_draws,
                                n_tune=n_tune,
                                n_chains=n_chains,
                                seed=seed,
                                targ_acpt=targ_acpt,
                            )

                        _append_metrics(
                            trace=trace,
                            metrics=metrics,
                            var_1=var_1,
                            var_2=var_2,
                            prior=prior,
                            i=i,
                            params=params,
                            n_rel_params=n_rel_params,
                            irrel_params_idxs=irrel_params_idxs,
                            hdi=hdi,
                            dim=dim,
                        )
    else:
        if n_vars is not None and frac_rel_vars is not None:
            # Generate parameters and randomly set a some to zero
            params = lognorm.rvs(
                s=0.5, size=[n_vars, n_repl], random_state=rng
            )

            n_rel_params = round(n_vars * frac_rel_vars, None)
            irrel_params_idxs = np.random.choice(
                n_vars, n_vars - n_rel_params, replace=False
            )
            params[irrel_params_idxs, :] = 0

            # Generate an intercept
            icpt = np.random.normal(loc=0, scale=1, size=n_repl)

            # Standard deviation of features
            std_vars = np.ones(shape=[n_vars, n_vars])

            for var_1 in vars_1:
                # Generate error term
                noise = np.random.normal(loc=0, scale=1, size=[var_1, n_repl])
                for i in range(0, n_repl):
                    for var_2 in vars_2:
                        # Generate covariance matrix
                        corr = np.ones(shape=[n_vars, n_vars]) * var_2

                        corr[np.diag_indices_from(corr)] = 1
                        covar = np.zeros(shape=[n_vars, n_vars])
                        for j in range(0, n_vars):
                            for k in range(j, n_vars):
                                covar[k, j] = (
                                    corr[k, j]
                                    * std_vars[k, i]
                                    * std_vars[j, i]
                                )
                        covar[np.triu_indices(n=n_vars, k=1)] = covar.T[
                            np.triu_indices(n=n_vars, k=1)
                        ]

                        # Generate random features
                        features = rng.multivariate_normal(
                            np.zeros(n_vars), covar, size=var_1
                        )
                        target = (
                            icpt[i]
                            + np.dot(features, params[:, i])
                            + noise[:, i]
                        )

                        prior_mdls = PriorMdls(
                            target=target, features=features
                        )

                        for prior in priors:
                            if prior == "BAL":
                                trace = run_sampler(
                                    model=prior_mdls.create_mdl_bal(),
                                    n_draws=n_draws,
                                    n_tune=n_tune,
                                    n_chains=n_chains,
                                    seed=seed,
                                    targ_acpt=targ_acpt,
                                )
                            elif prior == "DHS":
                                trace = run_sampler(
                                    model=prior_mdls.create_mdl_dhs(),
                                    n_draws=n_draws,
                                    n_tune=n_tune,
                                    n_chains=n_chains,
                                    seed=seed,
                                    targ_acpt=targ_acpt,
                                )
                            elif prior == "DL":
                                trace = run_sampler(
                                    model=prior_mdls.create_mdl_dl(),
                                    n_draws=n_draws,
                                    n_tune=n_tune,
                                    n_chains=n_chains,
                                    seed=seed,
                                    targ_acpt=targ_acpt,
                                )
                            elif prior == "HS":
                                trace = run_sampler(
                                    model=prior_mdls.create_mdl_hs(),
                                    n_draws=n_draws,
                                    n_tune=n_tune,
                                    n_chains=n_chains,
                                    seed=seed,
                                    targ_acpt=targ_acpt,
                                )
                            elif prior == "RHS":
                                trace = run_sampler(
                                    model=prior_mdls.create_mdl_rhs(),
                                    n_draws=n_draws,
                                    n_tune=n_tune,
                                    n_chains=n_chains,
                                    seed=seed,
                                    targ_acpt=targ_acpt,
                                )
                            else:
                                trace = run_sampler(
                                    model=prior_mdls.create_mdl_spsl(),
                                    n_draws=n_draws,
                                    n_tune=n_tune,
                                    n_chains=n_chains,
                                    seed=seed,
                                    targ_acpt=targ_acpt,
                                )

                            _append_metrics(
                                trace=trace,
                                metrics=metrics,
                                var_1=var_1,
                                var_2=var_2,
                                prior=prior,
                                i=i,
                                params=params,
                                n_rel_params=n_rel_params,
                                irrel_params_idxs=irrel_params_idxs,
                                hdi=hdi,
                                n_vars=n_vars,
                            )

    loss_low, rel_std_loss_low = PerfMetrics.calc_avg_loss(
        data=metrics["Loss"][vars_1[0]]
    )
    loss_high, rel_std_loss_high = PerfMetrics.calc_avg_loss(
        data=metrics["Loss"][vars_1[1]]
    )

    std_median_low = PerfMetrics.calc_avg_std(
        data=metrics["StdMed"][vars_1[0]]
    )
    std_median_high = PerfMetrics.calc_avg_std(
        data=metrics["StdMed"][vars_1[1]]
    )

    if simul == 1:
        covrg_low = PerfMetrics.calc_avg_metric(
            data=metrics["Coverage"][vars_1[0]], n_repl=n_repl, n_vars=dim
        )
        covrg_high = PerfMetrics.calc_avg_metric(
            data=metrics["Coverage"][vars_1[1]], n_repl=n_repl, n_vars=dim
        )
        type_1_err_low = PerfMetrics.calc_avg_metric(
            data=metrics["Type1Err"][vars_1[0]],
            n_repl=n_repl,
            n_vars=dim * (1 - vars_1[0] / 100),
        )

        type_1_err_high = PerfMetrics.calc_avg_metric(
            data=metrics["Type1Err"][vars_1[1]],
            n_repl=n_repl,
            n_vars=dim * (1 - vars_1[1] / 100),
        )
    else:
        if n_vars is not None and frac_rel_vars is not None:
            covrg_low = PerfMetrics.calc_avg_metric(
                data=metrics["Coverage"][vars_1[0]],
                n_repl=n_repl,
                n_vars=n_vars,
            )
            covrg_high = PerfMetrics.calc_avg_metric(
                data=metrics["Coverage"][vars_1[1]],
                n_repl=n_repl,
                n_vars=n_vars,
            )
            type_1_err_low = PerfMetrics.calc_avg_metric(
                data=metrics["Type1Err"][vars_1[0]],
                n_repl=n_repl,
                n_vars=n_vars * (1 - frac_rel_vars),
            )

            type_1_err_high = PerfMetrics.calc_avg_metric(
                data=metrics["Type1Err"][vars_1[1]],
                n_repl=n_repl,
                n_vars=n_vars * (1 - frac_rel_vars),
            )

    convrg_low = PerfMetrics.calc_avg_metric(data=metrics["Rhat"][vars_1[0]])
    convrg_high = PerfMetrics.calc_avg_metric(data=metrics["Rhat"][vars_1[1]])

    runtime_low = PerfMetrics.calc_avg_metric(
        data=metrics["Minutes"][vars_1[0]]
    )

    runtime_high = PerfMetrics.calc_avg_metric(
        data=metrics["Minutes"][vars_1[1]]
    )

    diverg_low = PerfMetrics.calc_avg_metric(data=metrics["Diverg"][vars_1[0]])

    diverg_high = PerfMetrics.calc_avg_metric(
        data=metrics["Diverg"][vars_1[1]]
    )

    metrics_tot: DefaultDict[Any, Any] = defaultdict(lambda: defaultdict(dict))

    if simul == 1:
        tex_symb = "A"
    else:
        tex_symb = r"\rho"

    for i, var_2 in enumerate(
        dict(
            zip(
                vars_2,
                [
                    rf"${tex_symb}={vars_2[0]}$",
                    rf"${tex_symb}={vars_2[1]}$",
                    rf"${tex_symb}={vars_2[2]}$",
                ],
            )
        ).values()
    ):
        _append_to_dict(
            metrics_tot=metrics_tot,
            dict_var="metrics_low_p1",
            var_2=var_2,
            keys=["Loss", "StdLoss", "StdMedian", "Coverage"],
            vals=[loss_low, rel_std_loss_low, std_median_low, covrg_low],
            vars_2=vars_2,
            i=i,
        )

        _append_to_dict(
            metrics_tot=metrics_tot,
            dict_var="metrics_high_p1",
            var_2=var_2,
            keys=["Loss", "StdLoss", "StdMedian", "Coverage"],
            vals=[loss_high, rel_std_loss_high, std_median_high, covrg_high],
            vars_2=vars_2,
            i=i,
        )

        _append_to_dict(
            metrics_tot=metrics_tot,
            dict_var="metrics_low_p2",
            var_2=var_2,
            keys=["TypeIErr", "Rhat", "Minutes", "Divergences"],
            vals=[type_1_err_low, convrg_low, runtime_low, diverg_low],
            vars_2=vars_2,
            i=i,
        )

        _append_to_dict(
            metrics_tot=metrics_tot,
            dict_var="metrics_high_p2",
            var_2=var_2,
            keys=["TypeIErr", "Rhat", "Minutes", "Divergences"],
            vals=[type_1_err_high, convrg_high, runtime_high, diverg_high],
            vars_2=vars_2,
            i=i,
        )

    df_metrics_low_p1 = _create_df_metrics(
        dict_var=metrics_tot["metrics_low_p1"],
        simul=simul,
        vars_2=vars_2,
    )

    df_metrics_high_p1 = _create_df_metrics(
        dict_var=metrics_tot["metrics_high_p1"], simul=simul, vars_2=vars_2
    )

    df_metrics_low_p2 = _create_df_metrics(
        dict_var=metrics_tot["metrics_low_p2"], simul=simul, vars_2=vars_2
    )

    df_metrics_high_p2 = _create_df_metrics(
        dict_var=metrics_tot["metrics_high_p2"], simul=simul, vars_2=vars_2
    )

    df_metrics_p1 = pd.concat([df_metrics_low_p1, df_metrics_high_p1], axis=1)
    df_metrics_p2 = pd.concat([df_metrics_low_p2, df_metrics_high_p2], axis=1)

    return loss_low, loss_high, df_metrics_p1, df_metrics_p2
