"""
Provide aligned basic arithmetic ops.

Simple arithmetic operations :py:func:`add`, :py:func:`divide`, :py:func:`multiply` and
:py:func:`subtract` which allow setting the standard how="outer" alignment that pandas
uses by default.

In practice, this means if dataframes do not share the same axes one can choose to get
the results for only the items index items existing in both indices (``how="inner"``) or
whether to prefer the axis from the first (``how="left"``) or the right (``how="right``)
operand.

See also
--------
pandas.DataFrame.align
"""

from typing import Any, Mapping, Tuple

from pandas import DataFrame, Series

from .core import Data


def _needs_axis(df: Data, other: Data) -> bool:
    return (isinstance(df, DataFrame) and isinstance(other, Series)) or (
        isinstance(df, Series) and isinstance(other, DataFrame)
    )


def _prepare_op(
    df: Data, other: Data, kwargs: Mapping[str, Any]
) -> Tuple[Data, Data, Mapping[str, Any]]:
    kwargs.setdefault("copy", True)
    if _needs_axis(df, other):
        kwargs.setdefault("axis", 0)
    df, other = df.align(other, **kwargs)
    return df, other, kwargs


def add(df: Data, other: Data, **align_kwargs: Any) -> Data:
    df, other, align_kwargs = _prepare_op(df, other, align_kwargs)
    return df.add(other, axis=align_kwargs.get("axis", 0))


def divide(df: Data, other: Data, **align_kwargs: Any) -> Data:
    df, other, align_kwargs = _prepare_op(df, other, align_kwargs)
    return df.div(other, axis=align_kwargs.get("axis", 0))


def multiply(df: Data, other: Data, **align_kwargs: Any) -> Data:
    df, other, align_kwargs = _prepare_op(df, other, align_kwargs)
    return df.mul(other, axis=align_kwargs.get("axis", 0))


def subtract(df: Data, other: Data, **align_kwargs: Any) -> Data:
    df, other, align_kwargs = _prepare_op(df, other, align_kwargs)
    return df.sub(other, axis=align_kwargs.get("axis", 0))
