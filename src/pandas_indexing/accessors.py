"""Registers convenience accessors into the ``pix`` namespace of each pandas
object.

Examples
--------
>>> df.pix.project(["model", "scenario"])

>>> df.index.pix.assign(unit="Mt CO2")

>>> df.pix.multiply(other, how="left")
"""

import warnings
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Sequence, Union

import pandas as pd
from deprecated.sphinx import deprecated
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.api.extensions import no_default

from . import arithmetics
from .core import (
    add_zeros_like,
    aggregatelevel,
    antijoin,
    assignlevel,
    describelevel,
    dropnalevel,
    extractlevel,
    fixindexna,
    formatlevel,
    isna,
    notna,
    projectlevel,
    semijoin,
    to_tidy,
    uniquelevel,
)
from .types import Axis, Data
from .units import convert_unit, dequantify, quantify
from .utils import doc


class _PixAccessor:
    """
    Convenience accessor for accessing `pandas-indexing` functions.
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __repr__(self):
        return describelevel(self._obj, as_str=True)

    @doc(assignlevel, df="")
    def assign(
        self,
        frame: Optional[Data] = None,
        order: bool = False,
        axis: Axis = 0,
        ignore_index: bool = False,
        **labels: Any,
    ) -> Union[DataFrame, Series, MultiIndex]:
        return assignlevel(
            self._obj,
            frame=frame,
            order=order,
            axis=axis,
            ignore_index=ignore_index,
            **labels,
        )

    @doc(extractlevel, index_or_data="")
    def extract(
        self,
        template: Optional[str] = None,
        *,
        keep: bool = False,
        dropna: bool = True,
        regex: bool = False,
        axis: Axis = 0,
        drop: Optional[bool] = None,
        optional: Optional[Sequence[str]] = None,
        **templates: str,
    ) -> Union[DataFrame, Series, Index]:
        if drop is not None:
            warnings.warn(
                "Argument `drop` is deprecated (use `keep` instead)", DeprecationWarning
            )
            keep = not drop

        return extractlevel(
            self._obj,
            template,
            keep=keep,
            dropna=dropna,
            regex=regex,
            axis=axis,
            optional=optional,
            **templates,
        )

    @doc(formatlevel, index_or_data="")
    def format(
        self, axis: Axis = 0, optional: Optional[Sequence[str]] = None, **templates: str
    ) -> Union[DataFrame, Series, Index]:
        return formatlevel(self._obj, axis=axis, optional=optional, **templates)

    @doc(uniquelevel, index_or_data="")
    def unique(
        self,
        levels: Union[str, Sequence[str], None],
        axis: Axis = 0,
    ) -> Index:
        return uniquelevel(self._obj, levels=levels, axis=axis)

    @doc(projectlevel, index_or_data="")
    def project(
        self,
        levels: Sequence[str],
        axis: Axis = 0,
    ) -> Union[DataFrame, Series, Index]:
        return projectlevel(self._obj, levels=levels, axis=axis)

    @doc(notna, index_or_data="")
    def notna(
        self,
        subset: Optional[Sequence[str]] = None,
        how: Literal["any", "all"] = "any",
        axis: Axis = 0,
    ):
        return notna(self._obj, subset=subset, how=how, axis=axis)

    @doc(isna, index_or_data="")
    def isna(
        self,
        subset: Optional[Sequence[str]] = None,
        how: Literal["any", "all"] = "any",
        axis: Axis = 0,
    ):
        return isna(self._obj, subset=subset, how=how, axis=axis)

    @doc(dropnalevel, index_or_data="")
    def dropna(
        self,
        subset: Optional[Sequence[str]] = None,
        how: Literal["any", "all"] = "any",
        axis: Axis = 0,
    ) -> Union[DataFrame, Series, Index]:
        return dropnalevel(self._obj, subset=subset, how=how, axis=axis)

    @doc(fixindexna, index_or_data="")
    def fixna(
        self,
        axis: Axis = 0,
    ) -> Union[DataFrame, Series, Index]:
        return fixindexna(self._obj, axis=axis)

    @doc(antijoin, index_or_data="")
    def antijoin(self, other: Index, *, axis: Axis = 0):
        return antijoin(self._obj, other, axis=axis)


class _DataPixAccessor(_PixAccessor):
    @doc(semijoin, frame_or_series="")
    def semijoin(
        self,
        other: Index,
        *,
        how: Literal["left", "right", "inner", "outer"] = "left",
        level: Union[str, int, None] = None,
        sort: bool = False,
        axis: Axis = 0,
        fill_value: Any = no_default,
        fail_on_reorder: bool = False,
    ) -> Union[DataFrame, Series]:
        return semijoin(
            self._obj,
            other,
            how=how,
            level=level,
            sort=sort,
            axis=axis,
            fill_value=fill_value,
            fail_on_reorder=fail_on_reorder,
        )

    @doc(quantify, data="", example_call="s.pix.quantify()")
    def quantify(
        self,
        level: str = "unit",
        unit: Optional[str] = None,
        axis: Axis = 0,
        copy: bool = False,
    ):
        return quantify(self._obj, level=level, unit=unit, axis=axis, copy=copy)

    def dequantify(self, level: str = "unit", axis: Axis = 0, copy: bool = False):
        return dequantify(self._obj, level=level, axis=axis, copy=copy)

    @doc(
        convert_unit,
        data="",
        convert_unit_s_km='s.pix.convert_unit("km")',
        convert_unit_s_m_to_km='s.pix.convert_unit({"m": "km"})',
    )
    def convert_unit(
        self,
        unit: Union[str, Mapping[str, str], Callable[[str], str]],
        level: Optional[str] = "unit",
        axis: Axis = 0,
    ):
        return convert_unit(self._obj, unit=unit, level=level, axis=axis)

    @doc(to_tidy, data="")
    def to_tidy(
        self,
        meta: Optional[DataFrame] = None,
        value_name: Optional[str] = "value",
        columns: Optional[str] = "year",
    ):
        return to_tidy(self._obj, meta=meta, value_name=value_name, columns=columns)

    @doc(aggregatelevel, data="")
    def aggregate(
        self,
        agg_func: str = "sum",
        axis: Axis = 0,
        dropna: bool = True,
        mode: Literal["replace", "append", "return"] = "replace",
        **levels: Dict[str, Sequence[Any]],
    ):
        return aggregatelevel(
            self._obj, agg_func=agg_func, axis=axis, dropna=dropna, mode=mode, **levels
        )

    @doc(add_zeros_like, data="")
    def add_zeros_like(
        self,
        reference: Union[MultiIndex, DataFrame, Series],
        /,
        derive: Optional[Dict[str, MultiIndex]] = None,
        **levels: Sequence[str],
    ):
        return add_zeros_like(self._obj, reference=reference, derive=derive, **levels)


def _create_forward_binop(op):
    def forward_binop(
        self,
        other: Data,
        assign: Optional[Dict[str, Any]] = None,
        axis: Optional[Axis] = None,
        **align_kwargs: Any,
    ):
        return getattr(arithmetics, op)(
            self._obj, other, assign=assign, axis=axis, **align_kwargs
        )

    return forward_binop


def _create_forward_unitbinop(op):
    def forward_unitbinop(
        self,
        other: Data,
        level: str = "unit",
        assign: Optional[Dict[str, Any]] = None,
        axis: Optional[Axis] = None,
        **align_kwargs: Any,
    ):
        return getattr(arithmetics, f"unit{op}")(
            self._obj, other, level=level, assign=assign, axis=axis, **align_kwargs
        )

    return forward_unitbinop


for op in arithmetics.ARITHMETIC_BINOPS:
    forward_binop = _create_forward_binop(op)
    setattr(_DataPixAccessor, op, forward_binop)
    for alt in arithmetics.ALTERNATIVE_NAMES.get(op, []):
        setattr(_DataPixAccessor, alt, forward_binop)

for op in arithmetics.ARITHMETIC_UNITBINOPS:
    forward_unitbinop = _create_forward_unitbinop(op)
    setattr(_DataPixAccessor, f"unit{op}", forward_unitbinop)
    for alt in arithmetics.ALTERNATIVE_NAMES.get(op, []):
        setattr(_DataPixAccessor, f"unit{alt}", forward_unitbinop)


@pd.api.extensions.register_dataframe_accessor("pix")
class DataFramePixAccessor(_DataPixAccessor):
    pass


@pd.api.extensions.register_series_accessor("pix")
class SeriesPixAccessor(_DataPixAccessor):
    pass


@pd.api.extensions.register_index_accessor("pix")
class IndexPixAccessor(_PixAccessor):
    pass


use_pix_instead = deprecated(
    reason="Use the new name ``df.pix`` of the accessor", version="0.2.9"
)


@pd.api.extensions.register_dataframe_accessor("idx")
@use_pix_instead
class DataFrameIdxAccessor(_DataPixAccessor):
    pass


@pd.api.extensions.register_series_accessor("idx")
@use_pix_instead
class SeriesIdxAccessor(_DataPixAccessor):
    pass


@pd.api.extensions.register_index_accessor("idx")
@use_pix_instead
class IndexIdxAccessor(_PixAccessor):
    pass
