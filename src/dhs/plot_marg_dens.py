#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script creates the marginal densities of various prior
distributions.
"""

# Imports

import json
import warnings

import hydra
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.core.config_store import ConfigStore

from dhs.helpers.base import PlotParams
from dhs.utils import DHSConfig, Logger, MargDens

# Hydra Setup

cs = ConfigStore.instance()
cs.store(name="dhs_config", node=DHSConfig)

# Function


@hydra.main(config_path="conf", config_name="config", version_base=None)
def plot_marg_dens(cfg: DHSConfig) -> None:
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
        kind="fig_big", fig_size=(6.4, 2.8)
    )
    pylab.rcParams.update(plot_params)

    warnings.filterwarnings("ignore")

    # Data

    seq_start, seq_stop, seq_num = cfg.densities.seq
    seq = np.linspace(seq_start, seq_stop, seq_num)

    lap_dens = MargDens.create_lap(seq=seq, noise=cfg.densities.noise)

    horse = MargDens.create_hs(seq=seq)

    dir_lap = []
    for theta_j in seq:
        dir_lap.append(MargDens.create_dl(a=0.5, seq_j=theta_j))

    reg_horse = MargDens.stock_up_dens(
        func=MargDens.create_dhs_rhs,
        kind="rhs",
        n_samples=cfg.densities.samples,
        n_vars=cfg.densities.dim,
        n_obs=cfg.densities.n_obs,
        noise=cfg.densities.noise,
        trunc=cfg.densities.trunc,
        seed=cfg.general.seed,
        frac_rel_vars=0.5,
    )

    dir_horse = MargDens.stock_up_dens(
        func=MargDens.create_dhs_rhs,
        kind="dhs",
        n_samples=cfg.densities.samples,
        n_vars=cfg.densities.dim,
        n_obs=cfg.densities.n_obs,
        noise=cfg.densities.noise,
        trunc=cfg.densities.trunc,
        seed=cfg.general.seed,
        spars=0.5,
    )

    # Plotting

    fig, ax = plt.subplots(nrows=1, ncols=2)

    sns.lineplot(
        x=seq,
        y=lap_dens,
        ax=ax[0],
        label="DE",
        linestyle="dashed",
        legend=False,
        color=stata_colors[0],
    )

    sns.lineplot(
        x=seq,
        y=horse,
        ax=ax[0],
        label="HS",
        linestyle=(0, (5, 10)),
        legend=False,
        color=stata_colors[1],
    )

    sns.lineplot(
        x=seq,
        y=dir_lap,
        ax=ax[0],
        label="DL",
        linestyle="dotted",
        legend=False,
        color=stata_colors[2],
    )

    sns.kdeplot(
        data=reg_horse,
        ax=ax[0],
        label="RHS",
        bw_adjust=1,
        linestyle="dashdot",
        legend=False,
        color=stata_colors[3],
    )

    sns.kdeplot(
        data=dir_horse,
        ax=ax[0],
        label="DHS",
        bw_adjust=1,
        linestyle="solid",
        legend=False,
        color="black",
    )

    ax[0].set_xlim(left=-3.25, right=3.25)
    ax[0].set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax[0].set_ylim(bottom=0, top=0.9)
    ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax[0].set(
        xlabel=r"$\theta$", ylabel=r"$\pi(\theta)$", title="Central Region"
    )

    sns.lineplot(
        x=seq,
        y=lap_dens,
        ax=ax[1],
        linestyle="dashed",
        legend=False,
        color=stata_colors[0],
    )

    sns.lineplot(
        x=seq,
        y=horse,
        ax=ax[1],
        linestyle="dashdot",
        legend=False,
        color=stata_colors[1],
    )

    sns.lineplot(
        x=seq,
        y=dir_lap,
        ax=ax[1],
        linestyle="dotted",
        legend=False,
        color=stata_colors[2],
    )

    sns.kdeplot(
        data=reg_horse,
        ax=ax[1],
        bw_adjust=5,
        linestyle=(0, (5, 10)),
        legend=False,
        color=stata_colors[3],
    )

    sns.kdeplot(
        data=dir_horse,
        ax=ax[1],
        bw_adjust=5,
        linestyle="solid",
        legend=False,
        color="black",
    )

    ax[1].set_xlim(left=2.75, right=np.max(seq) + 0.25)
    ax[1].set_xticks([3, 5, 7, 9, 11, 13, 15])
    ax[1].set_ylim(bottom=0, top=0.013)
    ax[1].set_yticks([0, 0.003, 0.006, 0.009, 0.012])
    ax[1].set(xlabel=r"$\theta$", ylabel="", title="Tails")

    plt.tight_layout()

    handles, labels = [], []
    for ele in ax:
        for handle, label in zip(*ele.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.1),
    )

    plt.subplots_adjust(bottom=0.2)

    if cfg.general.save:
        plt.savefig(fname=f"{cfg.paths.figures}fig1.pdf", transparent=True)
    plt.show(block=False)

    # Execution Time and Log File Finish

    log.info(f"Execution time: {Logger.get_exec_time(start_time=t_start)}")


if __name__ == "__main__":
    plot_marg_dens()
