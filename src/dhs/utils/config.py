# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

import os
from dataclasses import dataclass
from pathlib import Path

# Classes


@dataclass
class Paths:
    acc: str
    figures: str
    tables: str
    data: str
    models: str

    def __post_init__(self):
        """
        Validate input paths.

        Validate the input paths to ensure they are valid file system paths.
        """
        for attr in self.__annotations__:
            path = getattr(self, attr)
            if not Path(path).is_dir():
                raise ValueError(f"Invalid path: {path}")

    def create_directories(self) -> None:
        """
        Create the directories if they do not exist.
        """
        os.makedirs(self.acc, exist_ok=True)
        os.makedirs(self.figures, exist_ok=True)
        os.makedirs(self.tables, exist_ok=True)
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.models, exist_ok=True)


@dataclass
class Files:
    clr_plt: str
    alc_data: str

    def __post_init__(self):
        if not isinstance(self.clr_plt, str):
            raise TypeError("clr_plt must be a string")
        if not isinstance(self.alc_data, str):
            raise TypeError("alc_data must be a string")


@dataclass
class General:
    seed: int
    priors: list
    n_repl: int
    save: bool
    main_scripts: list

    def __post_init__(self):
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.priors, list):
            raise TypeError("priors must be a list")
        if not isinstance(self.n_repl, int):
            raise TypeError("n_repl must be an integer")
        if not isinstance(self.save, bool):
            raise TypeError("save must be a boolean")
        if not isinstance(self.main_scripts, list):
            raise TypeError("main_scripts must be a list")


@dataclass
class Densities:
    samples: int
    n_obs: int
    dim: int
    noise: float
    trunc: float
    seq: list

    def __post_init__(self):
        if not isinstance(self.samples, int):
            raise TypeError("samples must be an integer")
        if not isinstance(self.n_obs, int):
            raise TypeError("n_obs must be an integer")
        if not isinstance(self.dim, int):
            raise TypeError("dim must be an integer")
        if not isinstance(self.noise, (float, int)):
            raise TypeError("noise must be a float or integer")
        if not isinstance(self.trunc, (float, int)):
            raise TypeError("trunc must be a float or integer")
        if not isinstance(self.seq, list):
            raise TypeError("seq must be a list")


@dataclass
class Simul1:
    dim: int
    spars_lev: list
    sig_stren: list

    def __post_init__(self):
        if not isinstance(self.dim, int):
            raise TypeError("dim must be an integer")
        if not isinstance(self.spars_lev, list):
            raise TypeError("spars_lev must be a list")
        if not isinstance(self.sig_stren, list):
            raise TypeError("sig_stren must be a list")


@dataclass
class Simul2:
    n_obs: list
    n_vars: int
    frac_rel_vars: float
    feat_corr: list

    def __post_init__(self):
        if not isinstance(self.n_obs, int):
            raise TypeError("n_obs must be an integer")
        if not isinstance(self.n_vars, int):
            raise TypeError("n_vars must be an integer")
        if not isinstance(self.frac_rel_vars, float):
            raise TypeError("frac_rel_vars must be a float")
        if not isinstance(self.feat_corr, list):
            raise TypeError("feat_corr must be a list")


@dataclass
class Hyperparams:
    spars_lev: float
    conc_rate: list

    def __post_init__(self):
        if not isinstance(self.spars_lev, float):
            raise TypeError("spars_lev must be a float")
        if not isinstance(self.conc_rate, list):
            raise TypeError("conc_rate must be a list")


@dataclass
class Estimation:
    hdi: float
    n_chains: int
    n_draws: int
    n_tune: int
    targ_acpt: float

    def __post_init__(self):
        if not isinstance(self.hdi, float):
            raise TypeError("hdi must be a float")
        if not isinstance(self.n_chains, int):
            raise TypeError("n_chains must be an integer")
        if not isinstance(self.n_draws, int):
            raise TypeError("n_draws must be an integer")
        if not isinstance(self.n_tune, int):
            raise TypeError("n_tune must be an integer")
        if not isinstance(self.targ_acpt, float):
            raise TypeError("targ_acpt must be a float")


@dataclass
class Plotting:
    base_siz: float
    leg_mkr_siz: float
    line_width: float
    mkr_siz: float
    ax_line_width: float
    xtick_maj_width: float
    ytick_maj_width: float
    xtick_maj_siz: float
    ytick_maj_siz: float
    font_family: str

    def __post_init__(self):
        if not isinstance(self.base_siz, float):
            raise TypeError("base_siz must be a float")
        if not isinstance(self.leg_mkr_siz, float):
            raise TypeError("leg_mkr_siz must be a float")
        if not isinstance(self.line_width, float):
            raise TypeError("line_width must be a float")
        if not isinstance(self.mkr_siz, float):
            raise TypeError("mkr_siz must be a float")
        if not isinstance(self.ax_line_width, float):
            raise TypeError("ax_line_width must be a float")
        if not isinstance(self.xtick_maj_width, float):
            raise TypeError("xtick_maj_width must be a float")
        if not isinstance(self.ytick_maj_width, float):
            raise TypeError("ytick_maj_width must be a float")
        if not isinstance(self.xtick_maj_siz, float):
            raise TypeError("xtick_maj_siz must be a float")
        if not isinstance(self.ytick_maj_siz, float):
            raise TypeError("ytick_maj_siz must be a float")
        if not isinstance(self.font_family, str):
            raise TypeError("font_family must be a string")


@dataclass
class Example:
    hdi: float
    n_draws: int
    n_tune: int
    n_chains: int
    targ_acpt: float

    def __post_init__(self):
        if not isinstance(self.hdi, float):
            raise TypeError("hdi must be a float")
        if not isinstance(self.n_draws, int):
            raise TypeError("n_draws must be an integer")
        if not isinstance(self.n_tune, int):
            raise TypeError("n_tune must be an integer")
        if not isinstance(self.n_chains, int):
            raise TypeError("n_chains must be an integer")
        if not isinstance(self.targ_acpt, float):
            raise TypeError("targ_acpt must be a float")


@dataclass
class DHSConfig:
    paths: Paths
    files: Files
    general: General
    densities: Densities
    first_simul: Simul1
    second_simul: Simul2
    hyperparams: Hyperparams
    estimation: Estimation
    plotting: Plotting
    example: Example
