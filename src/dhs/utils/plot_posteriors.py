#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

from typing import List

import arviz as az
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Function


def plot_posteriors(
    mdl_trace: az.InferenceData,
    n_states: int,
    li_vars_plot: list,
    li_cues: list,
    fig_path: str,
    fig_name: str,
    save: bool,
) -> None:
    """
    Plot posterior distributions.

    Parameters
    ----------
    mdl_trace : az.InferenceData
        Record of the sampling process.
    n_states : int
        Number of states.
    li_vars_plot : list
        List of variables to be plotted.
    li_cues : list
        List of emotional cues.
    fig_path : str
        Figure path.
    fig_name : str
        Figure name.
    save: bool
        Save objects.
    """
    li_params = ["beta_11", "beta_12"]
    li_cues = [
        col
        for col in li_cues
        if col.startswith(tuple(["Surprise", "Suspense"]))
    ]

    fig, ax = plt.subplots(
        nrows=len(li_vars_plot), ncols=1, sharex="all", sharey="none"
    )

    for p in range(0, len(li_params)):
        li_cues_p = [col for col in li_cues if col.startswith(li_vars_plot[p])]

        post_samples: List[float] = []
        n_chains = len(mdl_trace["posterior"][li_params[p]]["chain"])
        n_draws = len(mdl_trace["posterior"][li_params[p]]["draw"])

        for i in range(0, len(li_cues_p)):
            for s in range(0, n_states):
                post_samples.extend(
                    (
                        np.concatenate(
                            mdl_trace["posterior"][li_params[p]][:, :, s, i]
                        )
                    )
                )

        data = pd.DataFrame(data=post_samples, columns=["Coefficient"])
        data["State"] = list(
            np.repeat(np.arange(0, n_states), n_chains * n_draws)
        ) * len(li_cues_p)
        data["Lag"] = np.repeat(
            np.arange(0, len(li_cues_p)), n_chains * n_draws * n_states
        )

        sns.violinplot(
            data=data,
            x="Lag",
            y="Coefficient",
            hue="State",
            split=True,
            inner="quartile",
            ax=ax[p],
        )

        # Offset
        delta = 0.02

        for ii, item in enumerate(ax[p].collections):
            if isinstance(item, matplotlib.collections.PolyCollection):
                path, = item.get_paths()
                vertices = path.vertices

                # Shift x-coordinates of path
                if ii % 2:  # to right
                    vertices[:, 0] += delta
                else:  # to left
                    vertices[:, 0] -= delta

        for i, line in enumerate(ax[p].get_lines()):
            line.get_path().vertices[:, 0] += delta if i // 3 % 2 else - delta

        plt.setp(ax[p].collections, alpha=0.9)
        ax[p].set_title(label=li_vars_plot[p], fontstyle="italic")

        if p != len(li_vars_plot) - 1:
            ax[p].set_xlabel(xlabel="")

        for i, l in enumerate(ax[p].lines[1:]):
            if i % 3 != 0:
                l.set_visible(False)

        ax[p].lines[0].set_visible(False)
        ax[p].axhline(y=0, color="black", linestyle="solid", alpha=0.6)

    plt.tight_layout()
    if save:
        plt.savefig(fname=f"{fig_path}{fig_name}.pdf", transparent=True)
    plt.show(block=False)
