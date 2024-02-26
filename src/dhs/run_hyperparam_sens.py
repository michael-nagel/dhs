#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script investigates the sensitivity of the Dirichlet-horseshoe
prior by investigating the performance of the prior for different
choices of hyperparameters based on the normal means problem.
"""

# Imports

import json
from typing import Dict, List

import arviz as az
import hydra
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.core.config_store import ConfigStore

from dhs.helpers.base import PlotParams, PriorMdls
from dhs.utils import (
    DHSConfig,
    Logger,
    PerfMetrics,
    create_func_dict,
    run_sampler,
)

# Hydra Setup

cs = ConfigStore.instance()
cs.store(name="dhs_config", node=DHSConfig)

# Function


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_hyperparam_sens(cfg: DHSConfig) -> None:
    # Logging

    log = Logger.init_logger(name=__name__)
    t_start = Logger.get_time()

    # External Files

    with open(f"{cfg.paths.acc}{cfg.files.clr_plt}", "r") as f:
        stata_colors = json.load(f)

    # Cockpit

    np.random.seed(cfg.general.seed)

    plt.close("all")
    sns.set_theme(palette=stata_colors, style="ticks")

    plot_params = PlotParams().set_rc_params(
        kind="fig_small", fig_size=(6.4, 4.8)
    )
    pylab.rcParams.update(plot_params)

    # Estimation

    loss: Dict[List[int], Dict[List[int], List[float]]] = {
        conc_rate: {sig_stren: [] for sig_stren in cfg.first_simul.sig_stren}
        for conc_rate in cfg.hyperparams.conc_rate
    }

    # Error term
    white_noise = np.random.normal(
        loc=0, scale=1, size=[cfg.first_simul.dim, cfg.general.n_repl]
    )

    for conc_rate in cfg.hyperparams.conc_rate:
        for sig_stren in cfg.first_simul.sig_stren:
            # Parameter generation
            params = (
                np.ones(shape=[cfg.first_simul.dim, cfg.general.n_repl])
                * sig_stren
            )
            n_rel_params = int(
                cfg.first_simul.dim * cfg.hyperparams.spars_lev / 100
            )
            irrel_params_idxs = np.random.choice(
                cfg.first_simul.dim,
                cfg.first_simul.dim - n_rel_params,
                replace=False,
            )
            params[
                irrel_params_idxs, :
            ] = 0  # Randomly set a some parameters to zero

            target = (
                params + white_noise
            )  # Calculate target (the true parameter corrupted with noise)

            for i in range(0, cfg.general.n_repl):
                trace = run_sampler(
                    model=PriorMdls(target=target[:, i]).create_mdl_dhs(),
                    n_draws=cfg.estimation.n_draws,
                    n_tune=cfg.estimation.n_tune,
                    n_chains=cfg.estimation.n_chains,
                    seed=cfg.general.seed,
                    targ_acpt=cfg.estimation.targ_acpt,
                )

                # Summary stats
                summary = az.summary(
                    trace,
                    hdi_prob=cfg.estimation.hdi,
                    stat_funcs=create_func_dict(),
                    var_names=["theta"],
                )

                # Calculation and appendage of metrics
                loss[conc_rate][sig_stren].append(
                    ((summary["median"] - params[:, i]) ** 2).sum()
                )

    df_loss, _ = PerfMetrics.calc_avg_loss(data=loss)
    df_loss = df_loss.T

    if cfg.general.save:
        df_loss.to_hdf(
            path_or_buf=f"{cfg.paths.models}hyperparam_sens.h5",
            key="hyperparam_sens",
            mode="w",
            format="table",
        )

    # Plotting

    line_styles = ["dashed", "dashdot", "dotted", "solid"]
    markers = ["v", "^", "s", "o"]
    colors = [stata_colors[0], stata_colors[1], stata_colors[2], "black"]
    labels = [
        rf"$\nu = $ {ele}" if ele else r"$\nu \sim Beta(2, 8)$"
        for ele in cfg.hyperparams.conc_rate
    ]

    fig, ax = plt.subplots()

    for i in range(0, df_loss.shape[1]):
        ax.plot(
            df_loss.index,
            df_loss.iloc[:, i].values,
            label=labels[i],
            linestyle=line_styles[i],
            marker=markers[i],
            markerfacecolor="none",
            color=colors[i],
        )

        ax.set(xlabel="Signal Strength", ylabel="Loss")
        ax.set_xticks(df_loss.index)

    plt.tight_layout()

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=df_loss.shape[1],
        frameon=False,
        handletextpad=0.1,
        columnspacing=0.5,
        borderpad=0,
        borderaxespad=0,
    )

    plt.subplots_adjust(bottom=0.2)

    if cfg.general.save:
        plt.savefig(fname=f"{cfg.paths.figures}fig5.pdf", transparent=True)
    plt.show(block=False)

    # Execution Time and Log File Finish

    log.info(f"Execution time: {Logger.get_exec_time(start_time=t_start)}")


if __name__ == "__main__":
    run_hyperparam_sens()
