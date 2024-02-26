#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

import math
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.stats import dirichlet, halfcauchy, invgamma, norm

# Class


class MargDens:
    def __init__(self) -> None:
        pass

    @staticmethod
    def create_cauchy(noise: float, seq: list) -> np.ndarray:
        """
        Create Cauchy density.

        This function creates the marginal density of the Cauchy prior.

        Parameters
        ----------
        noise : float
            Noise variance.
        seq : list
            Sequence (domain) of the figure.

        Returns
        -------
        np.ndarray
            Cauchy density.
        """
        return (1 / (np.pi * noise)) * (
            noise**2 / (np.array(seq) ** 2 + noise**2)
        )

    @staticmethod
    def create_dhs_rhs(
        kind: str,
        n_samples: int,
        n_vars: int,
        n_obs: int,
        noise: float,
        trunc: float,
        seed: int,
        spars: float | None = None,
        frac_rel_vars: float | None = None,
    ) -> np.ndarray:
        """
        Create Dirichlet-horseshoe or regularized horseshoe density.

        This function creates the marginal density of the
        Dirichlet-horseshoe prior or the regularized horseshoe prior.

        Parameters
        ----------
        kind: str
            Kind of density (dhs or rhs).
        n_samples : int
            Number of samples.
        n_vars : int
            Number of variables.
        n_obs : int
            Number of observations.
        noise : float
            Noise variance.
        trunc : float
             Truncation.
        seed : int
            Random seed.
        spars : float, optional
            Sparsity level.
        frac_rel_vars: float, optional
            Fraction of relevant variables.

        Returns
        -------
        np.ndarray
            Selected density.
        """
        rng = np.random.default_rng(seed)

        if kind == "dhs":
            joint_reg_ofst = dirichlet.rvs(
                alpha=np.ones(shape=n_vars) * (spars or 0),
                size=n_samples,
                random_state=rng,
            )

            joint_reg = joint_reg_ofst * n_vars
            n_rel_vars = (spars or 0) * n_vars
        else:
            n_rel_vars = (frac_rel_vars or 0) * n_vars

        glob_reg_scale = (n_rel_vars / (n_vars - n_rel_vars)) * (
            noise / np.sqrt(n_obs)
        )

        glob_reg = halfcauchy.rvs(
            loc=0, scale=glob_reg_scale, size=n_samples, random_state=rng
        )

        slab = invgamma.rvs(
            a=1, loc=0, scale=4, size=n_samples, random_state=rng
        )

        loc_reg_ofst = halfcauchy.rvs(
            loc=0, scale=1, size=n_samples, random_state=rng
        )

        loc_reg = (slab * loc_reg_ofst**2) / (
            slab + glob_reg**2 * loc_reg_ofst**2
        )

        if kind == "dhs":
            samples = norm.rvs(
                loc=0,
                scale=glob_reg * loc_reg**0.5 * joint_reg[:, 0],
                random_state=rng,
            )
        else:
            samples = norm.rvs(
                loc=0, scale=glob_reg * loc_reg**0.5, random_state=rng
            )

        return samples[np.abs(samples) <= trunc]

    @staticmethod
    def create_dl(a: float, seq_j: float) -> np.ndarray:
        """
        Create Dirichlet-Laplace density.

        This function creates the marginal density of the
        Dirichlet-Laplace prior.

        Parameters
        ----------
        a : float
            Concentration rate.
        seq_j : float
            Element of the sequence (domain) to be plotted.

        Returns
        -------
        np.ndarray
            Dirichlet-Laplace density.
        """
        v = 1 - a
        x = math.sqrt(2 * np.abs(seq_j))
        k = ((math.gamma(v + 0.5) * (2 * x) ** v) / math.sqrt(math.pi)) * quad(
            func=lambda t: math.cos(t) / (t**2 + x**2) ** (v + 0.5),
            a=0,
            b=np.inf,
        )[0]

        return (
            (1 / (2 ** ((1 + a) / 2) * math.gamma(a)))
            * np.abs(seq_j) ** ((a - 1) / 2)
        ) * k

    @staticmethod
    def create_hs(seq: list, bound: str = "upper") -> np.ndarray:
        """
        Create horseshoe density.

        This function creates the marginal density of the horseshoe
        prior.

        Parameters
        ----------
        seq : list
            The sequence (domain) to be plotted.
        bound : str, default upper
            Lower/upper bound.

        Returns
        -------
        np.ndarray
            Horseshoe density.
        """
        if bound == "upper":
            horse = (1 / np.sqrt(2 * np.pi**3)) * np.log(
                1 + 2 / np.array(seq) ** 2
            )
        else:
            print("Used bound = 'lower'")
            horse = ((1 / np.sqrt(2 * np.pi**3)) / 2) * np.log(
                1 + 4 / np.array(seq) ** 2
            )

        return horse

    @staticmethod
    def create_lap(seq: list, noise: float) -> np.ndarray:
        """
        Create Laplace density.

        This function creates the marginal density of the Laplace prior.

        Parameters
        ----------
        seq : list
            The sequence (domain) to be plotted.
        noise : float
            Noise variance.

        Returns
        -------
        np.ndarray
            Laplace density.
        """
        return (1 / (2 * noise)) * np.exp(-np.abs(np.array(seq)) / noise)

    @staticmethod
    def stock_up_dens(func: Callable, **kwargs) -> np.ndarray:
        """
        Draw additional samples from prior distribution.

        Parameters
        ----------
        func : Callable
            Name (path) of file.
        **kwargs
            These parameters will be passed to the function.

        Returns
        -------
        np.ndarray
            Additional samples.
        """
        n_samples = kwargs.get("n_samples", 1000)
        kwargs["n_samples"] = int(n_samples * 0.05)
        samples = func(**kwargs)

        while samples.shape[0] < n_samples:
            add_samples = func(**kwargs)
            samples = np.concatenate([samples, add_samples], axis=0)

        return samples[:n_samples]
