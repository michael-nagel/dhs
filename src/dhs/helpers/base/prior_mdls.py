#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

import numpy as np
import pandas as pd
import pymc as pm

from dhs.utils import _PmObj

# Class


class PriorMdls:
    def __init__(
        self,
        target: pd.Series,
        noise: float = 1,
        features=None,
        conc_rate=None,
    ) -> None:
        """
        Parameters
        ----------
        target : pd.Series
            Dependent variable.
        noise : float, default 1
            Noise variance.
        features : np.ndarray, optional
            Independent variables.
        conc_rate : float, optional
            Concentration rate.
        """
        self.target = target
        self.noise = noise
        self.features = features
        self.conc_rate = conc_rate
        if self.features is None:
            self.n_obs = 1
            self.dim = target.shape[0]
        else:
            self.n_obs = target.shape[0]
            self.dim = self.features.shape[1]

    def create_target_pred(
        self, param: pm.Distribution, icpt: pm.Distribution = None
    ) -> pm.Distribution:
        """
        Create the target prediction variable.

        Parameters
        ----------
        param : pm.Distribution
            Parameter distribution.
        icpt : pm.Distribution
            Intercept.

        Returns
        -------
        pm.Distribution
            Target prediction variable (model mean).
        """
        if self.features is None:
            return _PmObj.create_det(name="y_hat", var=param)
        else:
            return _PmObj.create_det(
                name="y_hat", var=pm.math.dot(self.features, param) + icpt
            )

    def create_mdl_bal(self) -> pm.Model:
        """
        Create a PyMC model using the Bayesian adaptive lasso prior.

        This function creates a random intercept model that uses the
        Bayesian adaptive lasso prior for estimation.

        Returns
        -------
        pm.Model
            Bayesian adaptive lasso PyMC model to be numerically
            estimated.
        """
        with pm.Model() as model:
            # Regularization parameter
            reg = _PmObj.create_halfcauchy(name="psi", shape=self.dim)

            # Parameters (uncentered representation)
            param_ofst = _PmObj.create_laplace(
                name="theta_tld", shape=self.dim
            )

            param = _PmObj.create_det(
                name="theta", var=param_ofst * self.noise / reg
            )

            # Intercept and model mean
            if self.features is not None:
                icpt = _PmObj.create_normal(name="alpha")
                target_pred = self.create_target_pred(param, icpt)
            else:
                target_pred = self.create_target_pred(param)

            # Model likelihood
            _PmObj.create_normal(
                name="likelihood",
                mu=target_pred,
                sigma=self.noise,
                observed=self.target,
            )

            return model

    def create_mdl_dhs(self) -> pm.Model:
        """
        Create a PyMC model using the Dirichlet-horseshoe prior.

        This function creates a random intercept model that uses the
        Dirichlet-horseshoe prior for estimation.

        Returns
        -------
        pm.Model
            Dirichlet-horseshoe PyMC model to be estimated.
        """
        with pm.Model() as model:
            # Sparsity parameter for both concentration rate and fraction of
            # relevant variables
            if self.conc_rate is None:
                spars = _PmObj.create_beta(name="eta")
            else:
                spars = self.conc_rate

            # Number of relevant variables
            n_rel_vars = self.dim * spars

            # Joint regularization prior
            joint_reg_ofst = _PmObj.create_dirichlet(
                name="phi_tld", comp=self.dim, concen=spars
            )
            joint_reg = _PmObj.create_det(
                name="phi", var=self.dim * joint_reg_ofst
            )

            # Scale of global regularization prior
            glob_reg_scale = (n_rel_vars / (self.dim - n_rel_vars)) * (
                self.noise / np.sqrt(self.n_obs)
            )

            # Global regularization prior
            glob_reg = _PmObj.create_halfcauchy(
                name="tau", beta=glob_reg_scale
            )

            # Slab width
            slab = _PmObj.create_invgamma(name="c_sq")

            # Local regularization prior
            loc_reg_ofst = _PmObj.create_halfcauchy(
                name="psi_tld", shape=self.dim
            )
            loc_reg = _PmObj.create_det(
                name="psi",
                var=(
                    slab
                    * pm.math.sqr(loc_reg_ofst)
                    / (
                        slab
                        + pm.math.sqr(glob_reg) * pm.math.sqr(loc_reg_ofst)
                    )
                ),
            )

            # Regression coefficients (uncentered representation)
            param_ofst = _PmObj.create_normal(name="theta_tld", shape=self.dim)
            param = _PmObj.create_det(
                name="theta",
                var=param_ofst * joint_reg * pm.math.sqrt(loc_reg) * glob_reg,
            )

            # Intercept and model mean
            if self.features is not None:
                icpt = _PmObj.create_normal(name="alpha")
                target_pred = self.create_target_pred(param, icpt)
            else:
                target_pred = self.create_target_pred(param)

            # Model likelihood
            _PmObj.create_normal(
                name="likelihood",
                mu=target_pred,
                sigma=self.noise,
                observed=self.target,
            )

            return model

    def create_mdl_dl(self) -> pm.Model:
        """
        Create a PyMC model using the Dirichlet-Laplace prior.

        This function creates a random intercept model that uses the
        Dirichlet-Laplace prior for estimation.

        Returns
        -------
        pm.Model
            Dirichlet-Laplace PyMC model to be estimated.
        """
        with pm.Model() as model:
            # Joint regularization prior
            concen = 0.5
            joint_reg = _PmObj.create_dirichlet(
                name="phi", comp=self.dim, concen=concen
            )

            # Global regularization
            glob_reg = _PmObj.create_gamma(
                name="tau", alpha=self.dim * concen, beta=0.5
            )

            # Regression coefficients (uncentered representation)
            param_ofst = _PmObj.create_laplace(
                name="theta_tld", shape=self.dim
            )
            param = _PmObj.create_det(
                name="theta", var=param_ofst * joint_reg * glob_reg
            )

            # Intercept and model mean
            if self.features is not None:
                icpt = _PmObj.create_normal(name="alpha")
                target_pred = self.create_target_pred(param, icpt)
            else:
                target_pred = self.create_target_pred(param)

            # Model likelihood
            _PmObj.create_normal(
                name="likelihood",
                mu=target_pred,
                sigma=self.noise,
                observed=self.target,
            )

            return model

    def create_mdl_hs(self) -> pm.Model:
        """
        Create a PyMC model using the horseshoe prior.

        This function creates a random intercept model that uses the
        horseshoe prior for estimation.

        Returns
        -------
        pm.Model
            Horseshoe PyMC model to be estimated.
        """
        with pm.Model() as model:
            # Global regularization prior
            glob_reg = _PmObj.create_halfcauchy(name="tau", beta=self.noise)

            # Local regularization prior
            loc_reg = _PmObj.create_halfcauchy(name="psi", shape=self.dim)

            # Regression coefficients (uncentered representation)
            param_ofst = _PmObj.create_normal(name="theta_tld", shape=self.dim)
            param = _PmObj.create_det(
                name="theta", var=param_ofst * glob_reg * loc_reg
            )

            # Intercept and model mean
            if self.features is not None:
                icpt = _PmObj.create_normal(name="alpha")
                target_pred = self.create_target_pred(param, icpt)
            else:
                target_pred = self.create_target_pred(param)

            # Model likelihood
            _PmObj.create_normal(
                name="likelihood",
                mu=target_pred,
                sigma=self.noise,
                observed=self.target,
            )

            return model

    def create_mdl_rhs(self) -> pm.Model:
        """
        Create a PyMC model using the regularized horseshoe prior.

        This function creates a random intercept model that uses the
        regularized horseshoe prior for estimation.

        Returns
        -------
        pm.Model
            Regularized horseshoe PyMC model to be estimated.
        """
        with pm.Model() as model:
            # Fraction of relevant variables
            frac_rel_vars = _PmObj.create_beta(name="eta")

            # Number of relevant variables
            n_rel_vars = _PmObj.create_det(
                name="p_0", var=frac_rel_vars * self.dim
            )

            # Scale of the global regularization prior
            glob_reg_scale = (n_rel_vars / (self.dim - n_rel_vars)) * (
                self.noise / np.sqrt(self.n_obs)
            )

            # Global regularization prior
            glob_reg = _PmObj.create_halfcauchy(
                name="tau", beta=glob_reg_scale
            )

            # Slab width
            slab = _PmObj.create_invgamma(name="c_sq")

            # Local regularization prior
            loc_reg_ofst = _PmObj.create_halfcauchy(
                name="psi_tld", shape=self.dim
            )
            loc_reg = _PmObj.create_det(
                name="psi",
                var=(
                    slab
                    * pm.math.sqr(loc_reg_ofst)
                    / (
                        slab
                        + pm.math.sqr(glob_reg) * pm.math.sqr(loc_reg_ofst)
                    )
                ),
            )

            # Regression coefficients (uncentered representation)
            param_ofst = _PmObj.create_normal(name="theta_tld", shape=self.dim)
            param = _PmObj.create_det(
                name="theta", var=param_ofst * glob_reg * pm.math.sqrt(loc_reg)
            )

            # Intercept and model mean
            if self.features is not None:
                icpt = _PmObj.create_normal(name="alpha")
                target_pred = self.create_target_pred(param, icpt)
            else:
                target_pred = self.create_target_pred(param)

            # Model likelihood
            _PmObj.create_normal(
                name="likelihood",
                mu=target_pred,
                sigma=self.noise,
                observed=self.target,
            )

            return model

    def create_mdl_spsl(self) -> pm.Model:
        """
        Create a PyMC model using the spike-and-slab prior.

        This function creates a random intercept model that uses the
        spike-and-slab prior for estimation.

        Returns
        -------
        pm.Model
            Spike-and-slab PyMC model to be estimated.
        """
        with pm.Model() as model:
            # Slab width
            slab = _PmObj.create_invgamma(name="c_sq")

            # Continuous prior for variable relevance
            switch = _PmObj.create_beta(
                name="lambda", alpha=0.5, beta=0.5, shape=self.dim
            )

            # Parameters (uncentered representation)
            param_ofst = _PmObj.create_normal(name="theta_tld", shape=self.dim)
            param = _PmObj.create_det(
                name="theta", var=param_ofst * switch * pm.math.sqrt(slab)
            )

            # Intercept and model mean
            if self.features is not None:
                icpt = _PmObj.create_normal(name="alpha")
                target_pred = self.create_target_pred(param, icpt)
            else:
                target_pred = self.create_target_pred(param)

            # Model likelihood
            _PmObj.create_normal(
                name="likelihood",
                mu=target_pred,
                sigma=self.noise,
                observed=self.target,
            )

            return model
