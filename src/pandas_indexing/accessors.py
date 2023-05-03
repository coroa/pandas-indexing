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
    dropnalevel,
    projectlevel,
    semijoin,
    summarylevel,
    uniquelevel,
)


class _IdxAccessor:
    """
    Convenience accessor for accessing `pandas-indexing` functions.
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __repr__(self):
        return summarylevel(self._obj)

    def assign(
        self,
        frame: Optional[Data] = None,
        order: bool = False,
        axis: int = 0,
        **labels: Any,
    ) -> Union[DataFrame, Series, MultiIndex]:
        """
        Add or overwrite levels on a multiindex.

        Parameters
        ----------
        frame : Series|DataFrame, optional
            Additional labels
        order : list of str, optional
            Level names in desired order or False, by default False
        axis : int, optional
            Axis where to update multiindex, by default 0
        **labels
            Labels for each new index level

        Returns
        -------
        df
            Series or DataFrame with changed index or new MultiIndex
        """
        return assignlevel(self._obj, frame=frame, order=order, axis=axis, **labels)

    def unique(
        self,
        levels: Union[str, Sequence[str], None],
        axis: Union[int, Literal["index", "columns"]] = 0,
    ) -> Index:
        """
        Return unique index levels.

        Parameters
        ----------
        levels : str or Sequence[str], optional
            Names of levels to get unique values of
        axis : int, optional
            Axis of DataFrame to check on, by default 0

        Returns
        -------
        unique_index : Index

        See also
        --------
        pandas.Index.unique
        """
        return uniquelevel(self._obj, levels=levels, axis=axis)

    def project(
        self,
        levels: Sequence[str],
        axis: Union[int, str] = 0,
    ) -> Union[DataFrame, Series, Index]:
        """
        Project multiindex to given `levels`

        Drops all levels except the ones explicitly mentioned from a given multiindex
        or an axis of a series or a dataframe.

        Parameters
        ----------
        levels : Sequence[str]
            Names of levels to project on (to keep)
        axis : int, optional
            Axis of DataFrame to project, by default 0

        Returns
        -------
        index_or_series : Index|MultiIndex|Series|DataFrame

        See also
        --------
        pandas.MultiIndex.droplevel
        pandas.Series.droplevel
        pandas.DataFrame.droplevel
        """
        return projectlevel(self._obj, levels=levels, axis=axis)

    def dropna(
        self,
        subset: Optional[Sequence[str]] = None,
        how: Literal["any", "all"] = "any",
        axis: Union[int, str] = 0,
    ) -> Union[DataFrame, Series, Index]:
        """
        Remove missing index values.

        Drops all index entries for which any or all (`how`) levels are
        undefined.

        Parameters
        ----------
        subset : Sequence[str], optional
            Names of levels on which to check for NA values
        how : "any" (default) or "all"
            Whether to remove an entry if all levels are NA only a single one
        axis : int, optional
            Axis of DataFrame to check on, by default 0

        Returns
        -------
        index_or_series : Index|MultiIndex|Series|DataFrame

        See also
        --------
        pandas.DataFrame.dropna
        pandas.Series.dropna
        pandas.Index.dropna
        """
        return dropnalevel(self._obj, subset=subset, how=how, axis=axis)


class _DataIdxAccessor(_IdxAccessor):
    def semijoin(
        self,
        other: Index,
        *,
        how: Literal["left", "right", "inner", "outer"] = "left",
        level: Union[str, int, None] = None,
        sort: bool = False,
        axis: Literal[0, 1] = 0,
    ) -> Union[DataFrame, Series]:
        """
        Semijoin `df_or_series` by index `other`

        Parameters
        ----------
        other : Index
            other index to join with
        how : {'left', 'right', 'inner', 'outer'}
            Join method to use
        level : None or str or int or
            single level on which to join, if not given join on all
        sort : bool, optional
            whether to sort the index
        axis : {0, 1}
            Axis on which to join

        Returns
        -------
        DataFrame or Series

        Raises
        ------
        TypeError
            If axis is not 0 or 1, or
            if df_or_series does not derive from DataFrame or Series

        See also
        --------
        pandas.Index.join
        """
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
