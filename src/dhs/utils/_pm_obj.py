#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

from typing import Any

import numpy as np
import pymc as pm

# Class


class _PmObj:
    def __init__(self) -> None:
        pass

    @staticmethod
    def create_normal(
        mu: float = 0, sigma: float = 1, *args, **kwargs
    ) -> pm.Distribution:
        """
        Create normal prior.

        Parameters
        ----------
        mu : float, default 0
            First Normal parameter.
        sigma : float, default 1
            second Normal parameter.
        *args : non-keyword arguments
            Parameters specific to the chosen distribution.
        **kwargs : keyword arguments
            Parameters specific to the chosen distribution.

        Returns
        -------
        distribution : pm.Distribution
            PyMC Distribution representing the specified distribution.
        """
        return pm.Normal(mu=mu, sigma=sigma, *args, **kwargs)

    @staticmethod
    def create_laplace(
        mu: float = 0, b: float = 1, *args, **kwargs
    ) -> pm.Distribution:
        """
        Create Laplace prior.

        Parameters
        ----------
        mu : float, default 0
            First Laplace parameter.
        b : float, default 1
            Second Laplace parameter.
        *args : non-keyword arguments
            Parameters specific to the chosen distribution.
        **kwargs : keyword arguments
            Parameters specific to the chosen distribution.

        Returns
        -------
        distribution : pm.Distribution
            PyMC Distribution representing the specified distribution.
        """
        return pm.Laplace(mu=mu, b=b, *args, **kwargs)

    @staticmethod
    def create_halfnormal(
        sigma: float = 1, *args, **kwargs
    ) -> pm.Distribution:
        """
        Create normal prior.

        Parameters
        ----------
        sigma : float, default 1
            HalfNormal parameter.
        *args : non-keyword arguments
            Parameters specific to the chosen distribution.
        **kwargs : keyword arguments
            Parameters specific to the chosen distribution.

        Returns
        -------
        distribution : pm.Distribution
            PyMC Distribution representing the specified distribution.
        """
        return pm.HalfNormal(sigma=sigma, *args, **kwargs)

    @staticmethod
    def create_halfcauchy(beta: float = 1, *args, **kwargs) -> pm.Distribution:
        """
        Create Half-Cauchy prior.

        Parameters
        ----------
        beta : float, default 1
            HalfCauchy parameter.
        *args : non-keyword arguments
            Parameters specific to the chosen distribution.
        **kwargs : keyword arguments
            Parameters specific to the chosen distribution.

        Returns
        -------
        distribution : pm.Distribution
            PyMC Distribution representing the specified distribution.
        """
        return pm.HalfCauchy(beta=beta, *args, **kwargs)

    @staticmethod
    def create_beta(
        alpha: float = 2, beta: float = 8, *args, **kwargs
    ) -> pm.Distribution:
        """
        Create beta prior.

        Parameters
        ----------
        alpha : float, default 2
            First Beta parameter.
        beta : float, default 8
            Second Beta parameter.
        *args : non-keyword arguments
            Parameters specific to the chosen distribution.
        **kwargs : keyword arguments
            Parameters specific to the chosen distribution.

        Returns
        -------
        distribution : pm.Distribution
            PyMC Distribution representing the specified distribution.
        """
        return pm.Beta(alpha=alpha, beta=beta, *args, **kwargs)

    @staticmethod
    def create_dirichlet(
        comp: int, concen, *args, **kwargs
    ) -> pm.Distribution:
        """
        Create Dirichlet prior.

        Parameters
        ----------
        comp : int
            Determines the number of components of the Dirichlet.
        concen :
            Concentration rate.
        *args : non-keyword arguments
            Parameters specific to the chosen distribution.
        **kwargs : keyword arguments
            Parameters specific to the chosen distribution.

        Returns
        -------
        distribution : pm.Distribution
            PyMC Distribution representing the specified distribution.
        """
        return pm.Dirichlet(a=np.ones(comp) * concen, *args, **kwargs)

    @staticmethod
    def create_gamma(
        alpha: float, beta: float, *args, **kwargs
    ) -> pm.Distribution:
        """
        Create gamma prior.

        Parameters
        ----------
        alpha : float
            First Gamma parameter
        beta : float
            Second Gamma parameter
        *args : non-keyword arguments
            Parameters specific to the chosen distribution.
        **kwargs : keyword arguments
            Parameters specific to the chosen distribution.

        Returns
        -------
        distribution : pm.Distribution
            PyMC Distribution representing the specified distribution.
        """
        return pm.Gamma(alpha=alpha, beta=beta, *args, **kwargs)

    @staticmethod
    def create_invgamma(
        alpha: float = 1, beta: float = 4, *args, **kwargs
    ) -> pm.Distribution:
        """
        Create inverse gamma prior.

        Parameters
        ----------
        alpha : float, default 1
            First Inverse Gamma parameter.
        beta : float, default 4
            Second Inverse Gamma parameter.
        *args : non-keyword arguments
            Parameters specific to the chosen distribution.
        **kwargs : keyword arguments
            Parameters specific to the chosen distribution.

        Returns
        -------
        distribution : pm.Distribution
            PyMC Distribution representing the specified distribution.
        """
        return pm.InverseGamma(alpha=alpha, beta=beta, *args, **kwargs)

    @staticmethod
    def create_det(var: Any, *args, **kwargs) -> pm.Distribution:
        """
        Create a Deterministic prior.

        Parameters
        ----------
        var : Any
            The variable to be used as the deterministic value.
        *args : non-keyword arguments
            Parameters specific to the chosen distribution.
        **kwargs : keyword arguments
            Parameters specific to the chosen distribution.

        Returns
        -------
        distribution : pm.Distribution
            PyMC Distribution representing the specified distribution.
        """
        return pm.Deterministic(var=var, *args, **kwargs)
