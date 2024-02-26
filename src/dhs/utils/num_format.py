#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

import numpy as np
import pandas as pd

# Function


class NumFormat:
    def __init__(self, my_format: str = "{:.2g}") -> None:
        self.my_format = my_format
        pass

    @staticmethod
    def format_num(
        in_val: int | float | str | bool | None,
    ) -> int | float | str | bool | None:
        """
        Format numbers according to the specified string operations.

        Parameters
        ----------
        in_val : int | float | str | bool | None
            Input number to be formatted.

        Returns
        -------
        int | float | str | bool | None
            Formatted number.
        """
        if not (
            isinstance(in_val, (int, float)) and not isinstance(in_val, bool)
        ) or (pd.isna(in_val)):
            out_val = in_val
        elif abs(in_val) >= 1000000:
            out_val = "{:,.{}{}}".format(in_val, 2, "e")
        elif int(in_val) == in_val:
            out_val = "{:,.{}{}}".format(in_val, 0, "f")
            if out_val == "-0":
                out_val = "0"
        elif (abs(in_val) > 100) | (
            (abs(in_val) > 10) & (round(in_val, 1) == round(in_val))
        ):
            out_val = "{:,.{}{}}".format(in_val, 0, "f")
        elif (abs(in_val) > 10) & (round(in_val, 1) != round(in_val)):
            out_val = "{:.{}{}}".format(in_val, 1, "f")
        elif abs(in_val) > 1:
            out_val = "{:.{}{}}".format(in_val, 2, "f")
        elif (abs(in_val) < 1) & (abs(in_val) >= 0.0001):
            out_val = "{:.{}{}}".format(in_val, 4, "f")
        else:
            out_val = "{:.{}{}}".format(in_val, 4, "g")
        return out_val

    @staticmethod
    def format_col(df: pd.Series) -> pd.Series:
        """
        Format a column of a DataFrame according to a specified format.

        Parameters
        ----------
        df : pd.Series
            DataFrame column to be formatted.

        Returns
        -------
        pd.Series
            Formatted DataFrame column.
        """
        df = df.apply(NumFormat.format_num)
        df_num = df.apply(pd.to_numeric, errors="ignore")
        if df_num.dtype == int or df_num.dtype == float:
            if (
                df_num.round().dropna().eq(df_num.dropna()).all()
                or (df_num.abs().dropna() >= 1000000).all()
            ):
                df = df
            elif (
                (df_num.abs().dropna() > 100)
                | (
                    (df_num.abs().dropna() > 10)
                    & (
                        df_num.abs().round(1).dropna()
                        == df_num.abs().round(0).dropna()
                    )
                )
            ).all():
                df = df.apply(pd.to_numeric, errors="coerce")
                df = df.map("{:.0f}".format)
            elif (df_num.abs().dropna() >= 10).all() & ~(
                df_num.abs().dropna().round(1)
                != df_num.abs().dropna().round(0)
            ).all():
                df = df.apply(pd.to_numeric, errors="coerce")
                df = df.map("{:.1f}".format)
            elif df_num.abs().dropna().between(1, 10, inclusive="left").all():
                df = df.apply(pd.to_numeric, errors="coerce")
                df = df.map("{:.2f}".format)
            elif df_num.abs().dropna().between(0, 1, inclusive="left").all():
                df = df.apply(pd.to_numeric, errors="coerce")
                df = df.map("{:.4f}".format)
            elif (df_num.abs() < 0.0001).all():
                df = df
        else:
            df = df

        return df.replace("nan", np.nan)

    def format_post(self, value):
        """
        Format a column of a DataFrame according to a specified format.

        This function attempts to convert the input value to a numeric
        value. If the conversion is successful, it formats the value to
        have the specific format. If the conversion fails (due to a
        ValueError or TypeError), it returns the original value
        unchanged.

        Parameters
        ----------
        value : _type_
            Input value to be formatted.

        Returns
        -------
        _type_
            Formatted value.
        """
        try:
            formatted_value = self.my_format.format(float(value))
            return formatted_value
        except (ValueError, TypeError):
            return value
