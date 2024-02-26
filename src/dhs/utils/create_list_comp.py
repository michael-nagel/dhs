# !/usr/bin/env python3
# -*- coding: utf-8 -*-


def create_list_comp(li: list, prefix: str) -> list:
    """
    Create list comprehension.

    Parameters
    ----------
    li : list
        Input list.
    prefix : str
        String prefix.

    Returns
    -------
    list
        List of variables.
    """
    return [col for col in li if col.startswith(prefix)]
