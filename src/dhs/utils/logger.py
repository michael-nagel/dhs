#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

import datetime as dt
import logging
import time

# Class


class Logger:
    """
    Start timing, initialize the logger and get execution time.
    """

    @staticmethod
    def get_time() -> float:
        """
        Start stop watch.

        Returns
        -------
        float
            Relative time.
        """
        return time.perf_counter()

    @staticmethod
    def init_logger(name) -> logging.Logger:
        """
        Initialize logger.

        Parameters
        ----------
        name : str | None
            Name of the logger. If no name is specified, return the root
            logger.

        Returns
        -------
        logging.RootLogger
            Logger with the specified name, creating it if necessary.
        """
        return logging.getLogger(name)

    @staticmethod
    def get_exec_time(start_time: float) -> dt.timedelta:
        """
        Get execution time.

        Parameters
        ----------
        start_time : float
            Start time in seconds.

        Returns
        -------
        dt.timedelta
            Execution time.
        """

        return dt.timedelta(seconds=time.perf_counter() - start_time)
