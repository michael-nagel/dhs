#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

import arviz as az
import pymc as pm
import pymc.sampling_jax as pmjax

# Function


def run_sampler(
    model: pm.Model,
    n_draws: int,
    n_tune: int,
    n_chains: int,
    seed: int,
    targ_acpt: float,
    *args,
    **kwargs,
) -> az.InferenceData:
    """
    Run the sampler.

    This function numerically estimates the defined PyMC model
    using Hamiltonian Monte Carlo methods (NUTS). The output
    (trace) object keeps track of the sampled values of each
    variable over the course of the sampling process and can be
    used to calculate summary statistics, to draw posterior
    distributions, and to diagnose convergence, for example.

    Parameters
    ----------
    model : pm.Model
        PyMC model to be estimated.
    n_draws : int
        Number of draws.
    n_tune : int
        Number of tuning samples (burn-in).
    n_chains : int
        Number of chains.
    seed : int
        Random seed.
    targ_acpt : float
        Target acceptance rate.
    *args : non-keyword arguments
        Parameters specific to the trace.
    **kwargs : keyword arguments
        Parameters specific to the trace.

    Returns
    -------
    arviz.InferenceData
        Record of the sampling process.
    """
    assert isinstance(model, pm.Model), "model must be of type pm.Model"
    assert (
        isinstance(n_draws, int) and n_draws > 0
    ), "n_draws must be a positive integer"
    assert (
        isinstance(n_tune, int) and n_tune >= 0
    ), "n_tune must be a non-negative integer"
    assert (
        isinstance(n_chains, int) and n_chains > 0
    ), "n_chains must be a positive integer"
    assert isinstance(seed, int), "seed must be of type int"
    assert (
        isinstance(targ_acpt, float) and 0 <= targ_acpt <= 1
    ), "targ_acpt must be a float between 0 and 1"

    with model:
        trace = pmjax.sample_numpyro_nuts(
            model=model,
            draws=n_draws,
            tune=n_tune,
            chains=n_chains,
            random_seed=seed,
            target_accept=targ_acpt,
            *args,
            **kwargs,
        )

    return trace
