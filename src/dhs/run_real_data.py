#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script investigates the (finite) sample performance of the
Dirichlet-horseshoe prior compared to other priors using a real-data
example. In this setting, we aim to estimate the effects of emotional
cues on alcohol consumption in a soccer stadium.
"""

# Imports

import json
import pickle
from collections import defaultdict
from typing import Any, DefaultDict

import hydra
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hydra.core.config_store import ConfigStore
from sklearn.preprocessing import StandardScaler

from dhs.helpers import create_mdl_alc_data
from dhs.helpers.base import PlotParams
from dhs.utils import (
    DHSConfig,
    Logger,
    create_list_comp,
    plot_posteriors,
    run_sampler,
)

# Hydra Setup

cs = ConfigStore.instance()
cs.store(name="dhs_config", node=DHSConfig)

# Function


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_real_data(cfg: DHSConfig) -> None:
    # Logging

    log = Logger.init_logger(name=__name__)
    t_start = Logger.get_time()

    # External Files

    with open(f"{cfg.paths.acc}{cfg.files.clr_plt}", "r") as f:
        stata_colors = json.load(f)

    with open(f"{cfg.paths.data}{cfg.files.alc_data}", "rb") as fp:
        dict_alc_data = pickle.load(fp)

    # Cockpit

    np.random.seed(cfg.general.seed)

    plt.close("all")
    sns.set_theme(palette=stata_colors, style="ticks")

    plot_params = PlotParams().set_rc_params(
        kind="fig_big", fig_size=(6.4, 4.8)
    )
    pylab.rcParams.update(plot_params)

    # Transformations

    data = dict_alc_data["data"].copy()
    li_cues = dict_alc_data["li_cues"]
    li_covars_1 = dict_alc_data["li_covars_1"]
    li_covars_2 = dict_alc_data["li_covars_2"]
    li_time_dum = dict_alc_data["li_time_dum"]
    li_states = dict_alc_data["li_states"]
    li_lag_target = dict_alc_data["li_lag_target"]

    # Demeaning of the dependent variable
    scaler_mean = StandardScaler(with_mean=True, with_std=False)
    data["BeerSls"] = scaler_mean.fit_transform(
        data["BeerSls"].values.reshape(-1, 1)
    )

    # Standardization of covariates
    scaler = StandardScaler(with_mean=True, with_std=True)
    li_vars_scale = (
        li_cues + li_covars_1 + li_covars_2 + li_lag_target + li_time_dum
    )

    data[li_vars_scale] = scaler.fit_transform(data[li_vars_scale])

    # Object Creation

    dict_data: DefaultDict[Any, Any] = defaultdict()

    # Cue specific state
    dict_data["state_sur"] = (
        data[create_list_comp(li=li_states, prefix="StateSur")]
        .astype(int)
        .values
    )

    # States
    dict_data["n_states"] = data["State_L0"].nunique()

    dict_data["states"] = (
        data[create_list_comp(li=li_states, prefix="State_L")]
        .astype("int")
        .values
    )

    dict_data["covars_1"] = data[li_covars_1].copy().values
    dict_data["n_covars_1"] = dict_data["covars_1"].shape[1]

    dict_data["time_dum"] = data[li_time_dum].copy().values
    dict_data["n_time_dum"] = dict_data["time_dum"].shape[1]

    dict_data["sur"] = data[
        create_list_comp(li=li_cues, prefix="Surprise")
    ].values
    dict_data["n_sur_vars"] = dict_data["sur"].shape[1]

    dict_data["sus"] = data[
        create_list_comp(li=li_cues, prefix="Suspense")
    ].values
    dict_data["n_sus_vars"] = dict_data["sus"].shape[1]

    dict_data["lag_target"] = data[[col for col in li_lag_target]].values
    dict_data["n_lag_target_vars"] = dict_data["lag_target"].shape[1]

    dict_data["covars_2"] = data.groupby("Matchup")[li_covars_2].first().values
    dict_data["n_covars_2"] = dict_data["covars_2"].shape[1]

    matchup_idxs, matchups = pd.factorize(data["Matchup"])

    dict_data["matchup_idxs"] = matchup_idxs

    dict_data["coords"] = {
        "matchup": matchups,
        "obs_id": np.arange(len(matchup_idxs)),
    }

    dict_data["target"] = data["BeerSls"].copy().values

    dict_data["n_lags"] = 9

    dict_data["n_matches"] = data.groupby("Matchup").ngroups

    # Estimation

    model = create_mdl_alc_data(dict_data=dict_data)

    trace = run_sampler(
        model=model,
        n_draws=cfg.estimation.n_draws,
        n_tune=cfg.estimation.n_tune,
        n_chains=cfg.estimation.n_chains,
        seed=cfg.general.seed,
        targ_acpt=cfg.estimation.targ_acpt,
        postprocessing_backend="cpu",
        chain_method="parallel",
        idata_kwargs={"log_likelihood": False},
    )

    if cfg.general.save:
        trace.to_netcdf(filename=f"{cfg.paths.models}trace_alc_data.nc")

    # Posterior Plotting

    plot_posteriors(
        mdl_trace=trace,
        n_states=dict_data["n_states"],
        li_vars_plot=["Surprise", "Suspense"],
        li_cues=li_cues,
        fig_path=cfg.paths.figures,
        fig_name="fig4",
        save=cfg.general.save,
    )

    # Execution Time and Log File Finish

    log.info(f"Execution time: {Logger.get_exec_time(start_time=t_start)}")


if __name__ == "__main__":
    run_real_data()
