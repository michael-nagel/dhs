# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script concurrently executes all scripts to replicate the paper.
"""

# Imports

import concurrent.futures
import subprocess
from typing import List

import hydra
from hydra.core.config_store import ConfigStore

from dhs.utils import DHSConfig, Logger

# Hydra Setup

cs = ConfigStore.instance()
cs.store(name="dhs_config", node=DHSConfig)

# Function


def run_script(script: str) -> None:
    """
    Execute a Python script using subprocess.run.

    Parameters
    ----------
    script : str
        The path to the script to be executed.

    Returns
    -------
    None
    """
    subprocess.run(["python", script])


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DHSConfig) -> None:
    # Initialize logger and start timer
    log = Logger.init_logger(name=__name__)
    t_start = Logger.get_time()

    # Get the list of scripts to be executed
    scripts: List[str] = cfg.general.main_scripts

    # Use ProcessPoolExecutor to run scripts in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit each script for execution
        futures = [executor.submit(run_script, script) for script in scripts]

        # Wait for all scripts to complete
        concurrent.futures.wait(futures)

    # Execution Time and Log File Finish

    log.info(f"Execution time: {Logger.get_exec_time(start_time=t_start)}")


if __name__ == "__main__":
    main()
