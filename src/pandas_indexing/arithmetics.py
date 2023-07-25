"""Provide aligned basic arithmetic ops.

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

import operator
from typing import Any, Dict, Optional

from pandas import DataFrame, Series
from pandas.core.ops import ARITHMETIC_BINOPS

from .core import assignlevel, uniquelevel
from .types import Axis, Data


ALTERNATIVE_NAMES = {
    "truediv": ["div", "divide"],
    "mul": ["multiply"],
    "sub": ["subtract"],
}


def _needs_axis(df: Data, other: Data) -> bool:
    return (isinstance(df, DataFrame) and isinstance(other, Series)) or (
        isinstance(df, Series) and isinstance(other, DataFrame)
    )


def _create_binop(op: str):
    def binop(
        df: Data,
        other: Data,
        assign: Optional[Dict[str, Any]] = None,
        axis: Optional[Axis] = None,
        **align_kwargs: Any,
    ):
        if assign is not None:
            df = assignlevel(df, **assign)
            other = assignlevel(other, **assign)

        align_kwargs.setdefault("copy", False)
        if _needs_axis(df, other):
            axis = 0
        if isinstance(df, Series) and isinstance(other, DataFrame):
            if align_kwargs.get("join") in ("left", "right"):
                align_kwargs["join"] = {"left": "right", "right": "left"}[
                    align_kwargs["join"]
                ]
            other, df = other.align(df, axis=axis, **align_kwargs)
        else:
            df, other = df.align(other, axis=axis, **align_kwargs)

        return getattr(df, op)(other, axis=axis)

    return binop


def _create_unitbinop(op, binop):
    def unitbinop(
        df: Data,
        other: Data,
        level: str = "unit",
        assign: Optional[Dict[str, Any]] = None,
        axis: Optional[Axis] = None,
        **align_kwargs: Any,
    ):
        df_unit = uniquelevel(df, level, axis=axis).item()
        other_unit = uniquelevel(other, level, axis=axis).item()

        import pint

        ur = pint.get_application_registry()
        quantity = getattr(operator, op)(ur(df_unit), ur(other_unit)).to_reduced_units()

        if assign is None:
            assign = dict()
        assign = {level: f"{quantity.units:~}"} | assign

        return binop(df, other, assign=assign, axis=axis, **align_kwargs) * quantity.m

    return unitbinop


for op in ARITHMETIC_BINOPS:
    binop = _create_binop(op)
    unitbinop = _create_unitbinop(op, binop)
    globals().update({op: binop, f"unit{op}": unitbinop})
    for alt in ALTERNATIVE_NAMES.get(op, []):
        globals().update({alt: binop, f"unit{alt}": unitbinop})
