.. Nagel et al. 2024 Replication Package documentation master file, created by
   sphinx-quickstart on Sun Dec 31 12:54:05 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Nagel et al. 2024 Replication Package Documentation
===================================================

This package produces the content of Nagel et al. (2024). Running the package 
directly will execute `__main__.py`, which simultaneously executes its 
submodules and with it all results:

.. code-block:: console

   $ python src/dhs

The submodules `plot_marg_dens.py`, `run_first_simul.py`, 
`run_hyperparam_sens.py`, `run_second_simul.py`, and `run_real_data.py` can 
either concurrently or sequentially be executed individually to reproduce only 
certain parts of the paper, because the structure is not a pipeline where 
submodules would depend on previous ones.

The random seed is globally set once for each program in the file 
`conf/config.yaml`, which contains parameters used by all programs. These 
variables can be modified by adding `.yaml files` like 
`conf/general/alt_seed.yaml`, which override the default configurations. 
However, there is no need to change any global variables.

.. warning::
    The current appearance settings of tables and figures may not perfectly 
    fit the output with changed seeds. 

The modules `helpers/create_dhs.py` and `run_dhs_example.py` offer general 
code for estimation with the Dirichlet-horseshoe prior. The programs serve to 
demonstrate the practical application of this prior in real-world scenarios. 
User can customize the syntax to suit their specific context. As a general 
practice, users typically do not need to modify the fundamental structure of 
the code. Modifications can be made through function inputs, such as the data 
or sampling attributes, or by adjusting the hyperparameters (e.g., the slab 
width of the specified priors). Experienced users may go beyond adjusting 
hyperparameters. For instance, they can introduce additional priors to 
accommodate random effects or alter the distribution from which the noise 
parameter is sampled. It is also possible to fix certain values (e.g., to 
scalars) instead of sampling them. The estimation is performed by PyMC, a 
Python library used for probabilistic programming. It allows users to perform 
Bayesian statistical modeling and perform Bayesian inference through 
Hamiltonian Monte Carlo (HMC) methods and other algorithms.

.. toctree::
   :caption: Details:

   main_modules
   list_objects
   logic_subpackages

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
