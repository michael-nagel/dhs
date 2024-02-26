#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def mod_tex_tab(tab: str) -> str:
    """
    Modify TeX table.

    Insert a midrule in a TeX table at a specific position.

    Parameters
    ----------
    tab : str
        TeX table.

    Returns
    -------
    str
        Modified TeX table.
    """
    lines = tab.split('\n')
    lines.insert(2, '\\midrule')
    return '\n'.join(lines)
