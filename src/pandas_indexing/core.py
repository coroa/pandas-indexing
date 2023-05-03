"""
Core module.
"""
from functools import reduce
from itertools import chain
from operator import and_, or_
from typing import Any, Literal, Optional, Sequence, TypeVar, Union

import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.core.indexes.frozen import FrozenList

from .utils import print_list


Data = Union[Series, DataFrame]
T = TypeVar("T", bound=Union[Index, DataFrame, Series])
S = TypeVar("S", bound=Union[DataFrame, Series])


def _assignlevel(
    index: Index, frame: Optional[Data] = None, order: bool = False, **labels: Any
) -> MultiIndex:
    if isinstance(index, MultiIndex):
        new_levels = list(index.levels)
        new_codes = list(index.codes)
        new_names = list(index.names)
    else:
        level = index.unique().dropna()
        new_levels = [level]
        new_codes = [level.get_indexer(index)]
        new_names = [index.name]

    if isinstance(frame, Series):
        frame = frame.to_frame()
    if isinstance(frame, DataFrame):
        labels = dict(chain(frame.items(), labels.items()))

    for level, lbls in labels.items():
        if np.isscalar(lbls):
            lvl = [lbls]
            cde = np.full(len(index), 0)
        else:
            lvl = Index(lbls).unique().dropna()
            cde = lvl.get_indexer(lbls)

        if level in new_names:
            i = new_names.index(level)
            new_levels[i] = lvl
            new_codes[i] = cde
        else:
            new_levels.append(lvl)
            new_codes.append(cde)
            new_names.append(level)

    new_index = MultiIndex(codes=new_codes, levels=new_levels, names=new_names)

    if order:
        new_index = new_index.reorder_levels(order)

    return new_index


def assignlevel(
    df: T,
    frame: Optional[Data] = None,
    order: bool = False,
    axis: int = 0,
    **labels: Any,
) -> T:
    """
    Add or overwrite levels on a multiindex.

    Parameters
    ----------
    df : Series|DataFrame|MultiIndex
        Series or DataFrame on which to change index or index to change
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
    if isinstance(df, Index):
        return _assignlevel(df, frame=frame, order=order, **labels)

    if isinstance(df, Series):
        index = df.index
    elif isinstance(df, DataFrame):
        index = df.index if axis == 0 else df.columns
    else:
        raise TypeError(
            f"assignlevel expects an Index, Series or DataFrame ({type(df)=})"
        )

    new_index = _assignlevel(index, frame=frame, order=order, **labels)

    return df.set_axis(new_index, axis=axis)


def _projectlevel(index: Index, levels: Sequence[str]) -> Index:
    levels = np.atleast_1d(levels)
    if len(levels) == 1:
        return index.get_level_values(levels[0])

    return index.droplevel(index.names.difference(levels)).reorder_levels(levels)


def projectlevel(
    index_or_series: T, levels: Sequence[str], axis: Union[int, str] = 0
) -> T:
    """
    Project multiindex to given `levels`

    Drops all levels except the ones explicitly mentioned from a given multiindex
    or an axis of a series or a dataframe.

    Parameters
    ----------
    index_or_series : MultiIndex|Series|DataFrame
        MultiIndex, Series or DataFrame to project
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
    if isinstance(index_or_series, Index):
        return _projectlevel(index_or_series, levels)

    index = index_or_series.index if axis in (0, "index") else index_or_series.columns
    return index_or_series.set_axis(_projectlevel(index, levels), axis=axis)


def _notna(
    index: Index,
    subset: Optional[Sequence[str]] = None,
    how: Literal["any", "all"] = "any",
) -> np.ndarray:
    index = ensure_multiindex(index)

    subset = index.names if subset is None else np.atleast_1d(subset)
    codes = [index.codes[index.names.index(n)] for n in subset]
    op = and_ if how == "any" else or_
    return reduce(op, [c != -1 for c in codes])


def dropnalevel(
    index_or_series: T,
    subset: Optional[Sequence[str]] = None,
    how: Literal["any", "all"] = "any",
    axis: Union[int, str] = 0,
) -> T:
    """
    Remove missing index values.

    Drops all index entries for which any or all (`how`) levels are
    undefined.

    Parameters
    ----------
    index_or_series : MultiIndex|Series|DataFrame
        MultiIndex, Series or DataFrame to project
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
    if isinstance(index_or_series, Index):
        return index_or_series[_notna(index_or_series, subset, how)]

    if axis in (0, "index"):
        return index_or_series.loc[_notna(index_or_series.index, subset, how)]

    return index_or_series.loc[:, _notna(index_or_series.columns, subset, how)]


def uniquelevel(
    index_or_data: Union[DataFrame, Series, Index],
    levels: Union[str, Sequence[str], None],
    axis: Union[int, Literal["index", "columns"]] = 0,
) -> Index:
    """
    Return unique index levels.

    Parameters
    ----------
    index_or_data : Index|Series|DataFrame
        Index, Series or DataFrame from which to get unique values
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
    if isinstance(index_or_data, (DataFrame, Series)):
        index_or_data = (
            index_or_data.index if axis in (0, "index") else index_or_data.columns
        )

    if levels is None or isinstance(levels, str):
        return index_or_data.unique(level=levels)

    levels = list(levels)
    if len(levels) == 1:
        return index_or_data.unique(level=levels[0])

    return projectlevel(index_or_data, levels).unique()


def _summarylevel(index: Index, n: int = 80) -> str:
    def name(l):
        return "<unnamed>" if l is None else l

    c1 = max(len(name(l)) for l in index.names) + 1
    c2 = n - c1 - 5
    return "\n".join(
        f" * {name(l):{c1}}: {print_list(index.unique(l), c2)}" for l in index.names
    )


def summarylevel(index_or_data: Union[DataFrame, Series, Index], n: int = 80):
    """
    Summarize index levels.

    Parameters
    ----------
    index_or_data : Index|Series|DataFrame
        Index, Series or DataFrame of which to summarize index levels
    n : int, default 80
        The maximum line length

    Returns
    -------
    summary : str

    See also
    --------
    pandas.DataFrame.describe
    """
    if isinstance(index_or_data, DataFrame):
        index_desc = _summarylevel(index_or_data.index, n=n)
        columns_desc = _summarylevel(index_or_data.columns, n=n)
        return f"Index:\n{index_desc}\n\nColumns:\n{columns_desc}"

    if isinstance(index_or_data, Series):
        index_or_data = index_or_data.index

    return f"Index:\n{_summarylevel(index_or_data, n=n)}"


def index_names(s, raise_on_index=False):
    if isinstance(s, (Series, DataFrame)):
        s = s.index

    if isinstance(s, MultiIndex):
        names = s.names
    elif isinstance(s, Index):
        if raise_on_index:
            exc = (
                raise_on_index
                if raise_on_index is not True
                else RuntimeError("s must have a MultiIndex")
            )
            raise exc
        names = FrozenList([s.name])
    else:
        names = FrozenList(s)

    return names


def _ensure_multiindex(s: Index) -> MultiIndex:
    if isinstance(s, MultiIndex):
        return s
    return MultiIndex.from_arrays([s], names=[s.name])


def ensure_multiindex(s: T) -> T:
    if isinstance(s, Index):
        return _ensure_multiindex(s)

    return s.set_axis(_ensure_multiindex(s.index))


def alignlevel(s, other):
    names = index_names(
        other,
        raise_on_index=RuntimeError(
            "For alignment, both indices must be of type MultiIndex"
            "; use alignlevels instead."
        ),
    )
    s_names = index_names(s)
    if names.difference(s_names):
        raise RuntimeError("Both indices need to be aligned; use alignlevels instead.")
    return ensure_multiindex(s).reorder_levels(names.union(s_names.difference(names)))


def alignlevels(l, r):
    l_names = index_names(l)
    r_names = index_names(r)

    l_and_r = FrozenList([l for l in l_names if l in r_names])
    l_but_not_r = l_names.difference(r_names)
    r_but_not_l = r_names.difference(l_names)

    return (
        ensure_multiindex(l).reorder_levels(l_and_r.union(l_but_not_r)),
        ensure_multiindex(r).reorder_levels(l_and_r.union(r_but_not_l)),
    )


def semijoin(
    df_or_series: S,
    other: Index,
    *,
    how: Literal["left", "right", "inner", "outer"] = "left",
    level: Union[str, int, None] = None,
    sort: bool = False,
    axis: Literal[0, 1] = 0,
) -> S:
    """
    Semijoin `df_or_series` by index `other`

    Parameters
    ----------
    df_or_series : DataFrame or Series
        data to be filtered
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

    axes = df_or_series.axes
    index = axes[axis]
    if level is None:
        index = ensure_multiindex(index)
        other = ensure_multiindex(other)

    new_index, left_idx, _ = index.join(
        other, how=how, level=level, return_indexers=True, sort=sort
    )

    cls = df_or_series.__class__
    axes[axis] = new_index

    if left_idx is None:
        return cls(df_or_series.values, *axes).__finalize__(df_or_series)

    if isinstance(df_or_series, DataFrame):
        if axis == 0:
            data = np.where(
                left_idx[:, np.newaxis] != -1, df_or_series.values[left_idx, :], np.nan
            )
        elif axis == 1:
            data = np.where(left_idx != -1, df_or_series.values[:, left_idx], np.nan)
    elif isinstance(df_or_series, Series):
        data = np.where(left_idx != -1, df_or_series.values[left_idx], np.nan)
    else:
        raise TypeError(
            f"df_or_series must derive from DataFrame or Series, but is {type(df_or_series)}"
        )

    return cls(data, *axes).__finalize__(df_or_series)
