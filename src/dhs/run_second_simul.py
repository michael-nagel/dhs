#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script investigates the (finite) sample performance of the
Dirichlet-horseshoe prior compared to other priors by simulating a
correlated features problem. In this setting, one aims to estimate a
D-dimensional parameter vector based on a NxK feature space corrupted
with iid standard normal noise.
"""

# Imports

import json

import hydra
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.core.config_store import ConfigStore

from dhs.helpers import create_simuls
from dhs.helpers.base import PlotParams
from dhs.utils import (
    DHSConfig,
    Logger,
    NumFormat,
    mod_tex_tab,
    plot_metrics,
    write_text_file,
)

# Hydra Setup

cs = ConfigStore.instance()
cs.store(name="dhs_config", node=DHSConfig)

# Function


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_second_simul(cfg: DHSConfig) -> None:
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

    # Simulation

    loss_low, loss_high, df_metrics_p1, df_metrics_p2 = create_simuls(
        simul=2,
        vars_1=cfg.second_simul.n_obs,
        vars_2=cfg.second_simul.feat_corr,
        priors=cfg.general.priors,
        n_repl=cfg.general.n_repl,
        seed=cfg.general.seed,
        n_draws=cfg.estimation.n_draws,
        n_tune=cfg.estimation.n_tune,
        n_chains=cfg.estimation.n_chains,
        targ_acpt=cfg.estimation.targ_acpt,
        hdi=cfg.estimation.hdi,
        n_vars=cfg.second_simul.n_vars,
        frac_rel_vars=cfg.second_simul.frac_rel_vars,
    )

    if cfg.general.save:
        for i, df in enumerate(
            [loss_low, loss_high, df_metrics_p1, df_metrics_p2]
        ):
            mode = "a" if i > 0 else "w"
            df = df.T if i > 1 else df
            df.to_hdf(
                path_or_buf=f"{cfg.paths.models}simul2_metrics.h5",
                key=f"df_{i}",
                mode=mode,
            )

    # Plotting

    plot_metrics(
        priors=cfg.general.priors,
        li_metrics=[loss_low, loss_high],
        metric_name="Loss",
        x_label="Predictor Correlations",
        fig_path=cfg.paths.figures,
        fig_name="fig3",
        save=cfg.general.save,
    )

    # TeX Files

    if cfg.general.save:
        write_text_file(
            file=f"{cfg.paths.tables}tab3.tex",
            body=mod_tex_tab(tab=df_metrics_p1.apply(NumFormat.format_col)
                             .style.format(na_rep="").to_latex())
        )

        write_text_file(
            file=f"{cfg.paths.tables}tab4.tex",
            body=mod_tex_tab(
                tab=df_metrics_p2.apply(NumFormat.format_col)
                .style.format(
                    formatter={
                        "Rhat": NumFormat(my_format="{:.2f}").format_post
                        }, na_rep="", escape="latex").to_latex())
        )

    # Execution Time and Log File Finish

    log.info(f"Execution time: {Logger.get_exec_time(start_time=t_start)}")


if __name__ == "__main__":
    run_second_simul()
