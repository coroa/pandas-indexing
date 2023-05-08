"""
Core module.
"""
from functools import reduce
from itertools import chain
from operator import and_, or_
from typing import Any, Literal, Optional, Sequence, TypeVar, Union

import numpy as np
from deprecated import deprecated
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.core.indexes.frozen import FrozenList

from .utils import Axis, doc, get_axis, print_list


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


@doc(
    df="""
    df : DataFrame, Series or Index
        Index, Series or DataFrame of which to change index levels\
    """
)
def assignlevel(
    df: T,
    frame: Optional[Data] = None,
    order: bool = False,
    axis: Axis = 0,
    **labels: Any,
) -> T:
    """
    Add or overwrite levels on a multiindex.

    Parameters
    ----------\
    {df}
    frame : Series|DataFrame, optional
        Additional labels
    order : list of str, optional
        Level names in desired order or False, by default False
    axis : {{0, 1, "index", "columns"}}, default 0
        Axis where to update multiindex
    **labels
        Labels for each new index level

    Returns
    -------
    df
        Series or DataFrame with changed index or new MultiIndex
    """
    if isinstance(df, Index):
        return _assignlevel(df, frame=frame, order=order, **labels)

    index = get_axis(df, axis)
    new_index = _assignlevel(index, frame=frame, order=order, **labels)
    return df.set_axis(new_index, axis=axis)


def _projectlevel(index: Index, levels: Sequence[str]) -> Index:
    levels = np.atleast_1d(levels)
    if len(levels) == 1:
        return index.get_level_values(levels[0])

    return index.droplevel(index.names.difference(levels)).reorder_levels(levels)


@doc(
    index_or_data="""
    index_or_data : DataFrame, Series or Index
        Index, Series or DataFrame to project\
    """
)
def projectlevel(index_or_data: T, levels: Sequence[str], axis: Axis = 0) -> T:
    """
    Project multiindex to given `levels`

    Drops all levels except the ones explicitly mentioned from a given multiindex
    or an axis of a series or a dataframe.

    Parameters
    ----------\
    {index_or_data}
    levels : Sequence[str]
        Names of levels to project on (to keep)
    axis : {{0, 1, "index", "columns"}}, default 0
        Axis of DataFrame to project

    Returns
    -------
    index_or_data : Index|MultiIndex|Series|DataFrame

    See also
    --------
    pandas.MultiIndex.droplevel
    pandas.Series.droplevel
    pandas.DataFrame.droplevel
    """
    if isinstance(index_or_data, Index):
        return _projectlevel(index_or_data, levels)

    index = get_axis(index_or_data.index, axis)
    return index_or_data.set_axis(_projectlevel(index, levels), axis=axis)


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


@doc(
    index_or_data="""
    index_or_data : DataFrame, Series or Index
        Index, Series or DataFrame of which to drop rows or columns\
    """
)
def dropnalevel(
    index_or_data: T,
    subset: Optional[Sequence[str]] = None,
    how: Literal["any", "all"] = "any",
    axis: Axis = 0,
) -> T:
    """
    Remove missing index values.

    Drops all index entries for which any or all (`how`) levels are
    undefined.

    Parameters
    ----------\
    {index_or_data}
    subset : Sequence[str], optional
        Names of levels on which to check for NA values
    how : "any" (default) or "all"
        Whether to remove an entry if all levels are NA only a single one
    axis : {{0, 1, "index", "columns"}}, default 0
        Axis of DataFrame to check on

    Returns
    -------
    index_or_data : Index|MultiIndex|Series|DataFrame

    See also
    --------
    pandas.DataFrame.dropna
    pandas.Series.dropna
    pandas.Index.dropna
    """
    if isinstance(index_or_data, Index):
        return index_or_data[_notna(index_or_data, subset, how)]

    if axis in (0, "index"):
        return index_or_data.loc[_notna(index_or_data.index, subset, how)]

    return index_or_data.loc[:, _notna(index_or_data.columns, subset, how)]


@doc(
    index_or_data="""
    index_or_data : DataFrame, Series or Index
        Index, Series or DataFrame of which to describe index levels\
    """
)
def uniquelevel(
    index_or_data: Union[DataFrame, Series, Index],
    levels: Union[str, Sequence[str], None],
    axis: Axis = 0,
) -> Index:
    """
    Return unique index levels.

    Parameters
    ----------\
    {index_or_data}
    levels : str or Sequence[str], optional
        Names of levels to get unique values of
    axis : {{0, 1, "index", "columns"}}, default 0
        Axis of DataFrame to check on

    Returns
    -------
    unique_index : Index

    See also
    --------
    pandas.Index.unique
    """
    index = get_axis(index_or_data, axis)

    if levels is None or isinstance(levels, str):
        return index.unique(level=levels)

    levels = list(levels)
    if len(levels) == 1:
        return index.unique(level=levels[0])

    return projectlevel(index, levels).unique()


def _describelevel(index: Index, n: int = 80) -> str:
    def name(l):
        return "<unnamed>" if l is None else l

    c1 = max(len(name(l)) for l in index.names) + 1
    c2 = n - c1 - 5
    return "\n".join(
        f" * {name(l):{c1}}: {print_list(index.unique(l), c2)}" for l in index.names
    )


@doc(
    index_or_data="""
    index_or_data : DataFrame, Series or Index
        Index, Series or DataFrame of which to describe index levels\
    """
)
def describelevel(
    index_or_data: Union[DataFrame, Series, Index], n: int = 80, as_str: bool = False
) -> Optional[str]:
    """
    Describe index levels.

    Parameters
    ----------\
    {index_or_data}
    n : int, default 80
        The maximum line length
    as_str : bool, default False
        Whether to return as string or print, instead

    Returns
    -------
    description : str, optional
        if print is False

    See also
    --------
    pandas.DataFrame.describe
    """
    if isinstance(index_or_data, DataFrame):
        index_desc = _describelevel(index_or_data.index, n=n)
        columns_desc = _describelevel(index_or_data.columns, n=n)
        description = f"Index:\n{index_desc}\n\nColumns:\n{columns_desc}"
    else:
        if isinstance(index_or_data, Series):
            index_or_data = index_or_data.index
        description = f"Index:\n{_describelevel(index_or_data, n=n)}"

    if as_str:
        return description

    print(description)


summarylevel = deprecated(
    describelevel, reason="Use describelevel instead", version="v0.2.5"
)


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


@doc(
    frame_or_series="""
    frame_or_series : DataFrame or Series
        data to be filtered\
    """
)
def semijoin(
    frame_or_series: S,
    other: Index,
    *,
    how: Literal["left", "right", "inner", "outer"] = "left",
    level: Union[str, int, None] = None,
    sort: bool = False,
    axis: Axis = 0,
) -> S:
    """
    Semijoin ``data`` by index ``other``

    Parameters
    ----------\
    {frame_or_series}
    other : Index
        other index to join with
    how : {{'left', 'right', 'inner', 'outer'}}
        Join method to use
    level : None or str or int or
        single level on which to join, if not given join on all
    sort : bool, optional
        whether to sort the index
    axis : {{0, 1, "index", "columns"}}
        Axis on which to join

    Returns
    -------
    DataFrame or Series

    Raises
    ------
    TypeError
        If axis is not 0 or 1, or
        if frame_or_series does not derive from DataFrame or Series

    See also
    --------
    pandas.Index.join
    """

    if isinstance(axis, str):
        if axis == "index":
            axis = 0
        elif axis == "columns":
            axis = 1
        else:
            raise ValueError(
                f"axis can only be one of 0, 1, 'index' or 'columns', not: {axis}"
            )

    axes = frame_or_series.axes
    index = axes[axis]
    if level is None:
        index = ensure_multiindex(index)
        other = ensure_multiindex(other)

    new_index, left_idx, _ = index.join(
        other, how=how, level=level, return_indexers=True, sort=sort
    )

    cls = frame_or_series.__class__
    axes[axis] = new_index

    if left_idx is None:
        return cls(frame_or_series.values, *axes).__finalize__(frame_or_series)

    if isinstance(frame_or_series, DataFrame):
        if axis == 0:
            data = np.where(
                left_idx[:, np.newaxis] != -1,
                frame_or_series.values[left_idx, :],
                np.nan,
            )
        elif axis == 1:
            data = np.where(left_idx != -1, frame_or_series.values[:, left_idx], np.nan)
    elif isinstance(frame_or_series, Series):
        data = np.where(left_idx != -1, frame_or_series.values[left_idx], np.nan)
    else:
        raise TypeError(
            f"frame_or_series must derive from DataFrame or Series, but is {type(frame_or_series)}"
        )

    return cls(data, *axes).__finalize__(frame_or_series)
