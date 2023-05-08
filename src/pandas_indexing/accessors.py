"""
Registers convenience accessors into the ``idx`` namespace of each pandas
object.

Usage
-----
>>> df.idx.project(["model", "scenario"])

>>> df.index.idx.assign(unit="Mt CO2")

>>> df.idx.multiply(other, how="left")
"""

from typing import Any, Literal, Optional, Sequence, Union

import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series

from . import arithmetics
from .core import (
    Data,
    assignlevel,
    describelevel,
    dropnalevel,
    extractlevel,
    formatlevel,
    isna,
    notna,
    projectlevel,
    semijoin,
    uniquelevel,
)
from .utils import Axis, doc


class _IdxAccessor:
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
        **labels: Any,
    ) -> Union[DataFrame, Series, MultiIndex]:
        return assignlevel(self._obj, frame=frame, order=order, axis=axis, **labels)

    @doc(extractlevel, index_or_data="")
    def extract(
        self, axis: Axis = 0, **templates: str
    ) -> Union[DataFrame, Series, Index]:
        return extractlevel(self._obj, axis=axis, **templates)

    @doc(formatlevel, index_or_data="")
    def format(
        self, axis: Axis = 0, **templates: str
    ) -> Union[DataFrame, Series, Index]:
        return formatlevel(self._obj, axis=axis, **templates)

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
        return ~isna(self._obj, subset=subset, how=how, axis=axis)

    @doc(dropnalevel, index_or_data="")
    def dropna(
        self,
        subset: Optional[Sequence[str]] = None,
        how: Literal["any", "all"] = "any",
        axis: Axis = 0,
    ) -> Union[DataFrame, Series, Index]:
        return dropnalevel(self._obj, subset=subset, how=how, axis=axis)


class _DataIdxAccessor(_IdxAccessor):
    @doc(semijoin, frame_or_series="")
    def semijoin(
        self,
        other: Index,
        *,
        how: Literal["left", "right", "inner", "outer"] = "left",
        level: Union[str, int, None] = None,
        sort: bool = False,
        axis: Axis = 0,
    ) -> Union[DataFrame, Series]:
        return semijoin(self._obj, other, how=how, level=level, sort=sort, axis=axis)

    def multiply(self, other, **align_kwds):
        return arithmetics.multiply(self._obj, other, **align_kwds)

    def divide(self, other, **align_kwds):
        return arithmetics.divide(self._obj, other, **align_kwds)

    def add(self, other, **align_kwds):
        return arithmetics.add(self._obj, other, **align_kwds)

    def subtract(self, other, **align_kwds):
        return arithmetics.subtract(self._obj, other, **align_kwds)


@pd.api.extensions.register_dataframe_accessor("idx")
class DataFrameIdxAccessor(_DataIdxAccessor):
    pass


@pd.api.extensions.register_series_accessor("idx")
class SeriesIdxAccessor(_DataIdxAccessor):
    pass


@pd.api.extensions.register_index_accessor("idx")
class IndexIdxAccessor(_IdxAccessor):
    pass
