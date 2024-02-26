#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

from dhs.utils import DHSConfig

# Hydra Setup

cs = ConfigStore.instance()
cs.store(name="dhs_config", node=DHSConfig)

# Class

with initialize(
    version_base=None, config_path="../../conf", job_name="plot_params"
):
    cfg = compose(config_name="config")


class PlotParams:
    font_scalings = {
        "xx-small": 0.579,
        "x-small": 0.694,
        "small": 0.833,
        "medium": 1.0,
        "large": 1.2,
        "x-large": 1.44,
        "xx-large": 1.728,
        "larger": 1.2,
        "smaller": 0.833,
        None: 1.0,
    }

    def __init__(self) -> None:
        self.medium = cfg.plotting.base_size
        self.legend_fontsize, self.legend_title_fontsize = (
            self.medium,
            self.medium,
        )
        self.axes_labelsize, self.axes_titlesize = (
            self.medium,
            self.medium * PlotParams.font_scalings["large"],
        )
        self.legend_markerscale = cfg.plotting.leg_mkr_size
        self.lines_linewidth = cfg.plotting.line_width
        self.lines_markersize = cfg.plotting.mkr_size
        self.axes_linewidth = cfg.plotting.ax_line_width
        self.xtick_labelsize, self.ytick_labelsize = (self.medium, self.medium)
        self.xtick_major_width, self.ytick_major_width = (
            cfg.plotting.xtick_maj_width,
            cfg.plotting.ytick_maj_width,
        )
        self.xtick_major_size, self.ytick_major_size = (
            cfg.plotting.xtick_maj_size,
            cfg.plotting.ytick_maj_size,
        )
        self.font_family = [cfg.plotting.font_family]

    def set_rc_params(self, kind: str, fig_size: tuple) -> dict:
        """
        Define plotting parameters.

        This function returns the plotting parameters for different
        plot styles.

        Parameters
        ----------
        kind : str
            Type of figure [fig_small, fig_medium, fig_big].
        fig_size : tuple
            Figure size.

        Returns
        -------
        dict
            Dictionary with parameters for plotting.
        """
        # 50%
        if kind == "fig_small":  # 50%, single plots
            factor = 1.0
        elif kind == "fig_medium":  # 75%, 2x2 plots
            factor = 0.5 / 0.75
        elif kind == "fig_big":  # 100%, 2x1 and 1x2 plots
            factor = 0.5
        else:
            raise ValueError("Invalid kind parameter: {}".format(kind))

        params = {
            "legend.fontsize": self.legend_fontsize * factor,
            "legend.title_fontsize": self.legend_title_fontsize * factor,
            "legend.markerscale": self.legend_markerscale,
            "figure.figsize": fig_size,
            "axes.labelsize": self.axes_labelsize * factor,
            "axes.titlesize": self.axes_titlesize * factor,
            "lines.linewidth": self.lines_linewidth * factor,
            "lines.markersize": self.lines_markersize * factor,
            "axes.linewidth": self.axes_linewidth * factor,
            "xtick.labelsize": self.xtick_labelsize * factor,
            "ytick.labelsize": self.ytick_labelsize * factor,
            "xtick.major.width": self.xtick_major_width * factor,
            "ytick.major.width": self.ytick_major_width * factor,
            "xtick.major.size": self.xtick_major_size * factor,
            "ytick.major.size": self.ytick_major_size * factor,
            "font.family": self.font_family,
        }

        return params
