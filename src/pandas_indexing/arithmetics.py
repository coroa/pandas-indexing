"""
Provide aligned basic arithmetic ops.

Simple arithmetic operations ``add``, ``divide``, ``multiply`` and ``subtract`` which
allow setting the standard how="outer" alignment that pandas uses by default.

In practice, this means if dataframes do not share the same axes one can choose to get
the results for only the items index items existing in both indices (``how="inner"``) or
whether to prefer the axis from the first (``how="left"``) or the right (``how="right``)
operand.

See also
--------
`pandas.DataFrame.align`
"""

from typing import Union

from pandas import DataFrame, Series


def _needs_axis(df: Union[Series, DataFrame], other: Union[Series, DataFrame]) -> bool:
    return (isinstance(df, DataFrame) and isinstance(other, Series)) or (
        isinstance(df, Series) and isinstance(other, DataFrame)
    )


def add(df, other, **align_kwds):
    axis = align_kwds.setdefault("axis", 0) if _needs_axis(df, other) else 0
    df, other = df.align(other, **align_kwds)
    return df.add(other, axis=axis)


def divide(df, other, **align_kwds):
    axis = align_kwds.setdefault("axis", 0) if _needs_axis(df, other) else 0
    df, other = df.align(other, **align_kwds)
    return df.div(other, axis=axis)


def multiply(df, other, **align_kwds):
    axis = align_kwds.setdefault("axis", 0) if _needs_axis(df, other) else 0
    df, other = df.align(other, **align_kwds)
    return df.mul(other, axis=axis)


def subtract(df, other, **align_kwds):
    axis = align_kwds.setdefault("axis", 0) if _needs_axis(df, other) else 0
    df, other = df.align(other, **align_kwds)
    return df.sub(other, axis=axis)
