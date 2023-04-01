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


def add(df, other, **align_kwds):
    df, other = df.align(other, **align_kwds)
    return df + other


def divide(df, other, **align_kwds):
    df, other = df.align(other, **align_kwds)
    return df / other


def multiply(df, other, **align_kwds):
    df, other = df.align(other, **align_kwds)
    return df * other


def subtract(df, other, **align_kwds):
    df, other = df.align(other, **align_kwds)
    return df - other
