#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script provides a general estimation example for the DHS prior.
"""

# Imports

import json

import arviz as az
import hydra
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.core.config_store import ConfigStore

from dhs.helpers import create_dhs
from dhs.helpers.base import PlotParams
from dhs.utils import DHSConfig, Logger, run_sampler

# Hydra Setup

cs = ConfigStore.instance()
cs.store(name="dhs_config", node=DHSConfig)

# Function


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_dhs_example(cfg: DHSConfig) -> None:
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

    # Estimation Example

    # Example data
    features = np.random.randint(low=0, high=10, size=[100, 5])
    icpt = 0.5
    slopes = np.array([0, 0.2, 0.3, 0, 0])
    noise = np.random.randn(100)
    target = np.dot(features, slopes) + icpt + noise

    # Estimation
    trace = run_sampler(
        model=create_dhs(target=target, features=features),
        seed=cfg.general.seed,
        n_draws=cfg.example.n_draws,
        n_tune=cfg.example.n_tune,
        n_chains=cfg.example.n_chains,
        targ_acpt=cfg.example.targ_acpt,
    )
    summary = az.summary(data=trace, hdi_prob=cfg.example.hdi)

    if cfg.general.save:
        summary.to_hdf(
            path_or_buf=f"{cfg.paths.models}res_dhs_example.h5",
            key="res_dhs_example",
            mode="w",
        )

    # Plotting

    az.plot_trace(
        data=trace,
        var_names=["alpha", "beta"],
        compact=False,
        divergences=None,
    )
    plt.show(block=False)

    # Execution Time and Log File Finish

    log.info(f"Execution time: {Logger.get_exec_time(start_time=t_start)}")


if __name__ == "__main__":
    run_dhs_example()
