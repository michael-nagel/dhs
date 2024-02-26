#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

import matplotlib.pyplot as plt

# Function


def plot_metrics(
    priors: list,
    li_metrics: list,
    metric_name: str,
    x_label: str,
    fig_name: str,
    fig_path: str,
    save: bool,
) -> None:
    """
    Plot performance metrics.

    Parameters
    ----------
    priors : list
        List of DataFrames.
    li_metrics : list
        Dataframe metrics.
    metric_name : str
        Metric names.
    x_label : str
        X-axis label.
    fig_path : str
        Figure path.
    fig_name : str
        Figure name.
    save : bool
        Save object.
    """
    line_styles = [
        "dashed",
        "dashdot",
        "dotted",
        (0, (5, 10)),
        (0, (1, 10)),
        "solid",
    ]
    markers = ["v", "^", "s", "d", "p", "o"]
    colors = ["C0", "C1", "C2", "C4", "C5", "black"]

    if x_label == "Signal Strength":
        li_title = ["$q_D=0.05$", "$q_D=0.1$"]
    else:
        li_title = ["$N=80$", "$N=200$"]

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex="col", sharey="none")

    for i, metric in enumerate(li_metrics):
        for j, col in enumerate(priors):
            ax[i].plot(
                metric.index,
                metric[col],
                label=priors[j],
                linestyle=line_styles[j],
                marker=markers[j],
                markerfacecolor="none",
                color=colors[j],
            )

        ax[i].set(xlabel=x_label)
        ax[i].set_title(label=li_title[i], fontstyle="italic")
        ax[i].set_xticks(metric.index)

        if i == 0:
            ax[i].set(ylabel=metric_name)

    plt.tight_layout()

    handles, labels = [], []
    for handle, label in zip(*ax[0].get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.08),
    )

    plt.subplots_adjust(bottom=0.2)

    if save:
        plt.savefig(fname=f"{fig_path}{fig_name}.pdf", transparent=True)
    plt.show(block=False)
