"""
core module
"""
from functools import reduce
from operator import and_
from typing import Any, Sequence, Union

import numpy as np
from deprecated import deprecated
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.core.indexes.frozen import FrozenList

from .utils import shell_pattern_to_regex


def _assignlevel(index: Index, order: bool = False, **labels: Any) -> MultiIndex:
    if isinstance(index, MultiIndex):
        new_levels = list(index.levels)
        new_codes = list(index.codes)
        new_names = list(index.names)
    else:
        level = index.unique().dropna()
        new_levels = [level]
        new_codes = [level.get_indexer(index)]
        new_names = [index.name]

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
    df: Union[Index, DataFrame, Series],
    order: bool = False,
    axis: int = 0,
    **labels: Any,
):
    """Add or overwrite levels on a multiindex

    Parameters
    ----------
    df : Series|DataFrame|MultiIndex
        Series or DataFrame on which to change index or index to change
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
        return _assignlevel(df, order=order, **labels)

    if isinstance(df, Series):
        index = df.index
    elif isinstance(df, DataFrame):
        index = df.index if axis == 0 else df.columns
    else:
        raise TypeError(
            f"assignlevel expects an Index, Series or DataFrame ({type(df)=})"
        )

    new_index = _assignlevel(index, order=order, **labels)

    return df.set_axis(new_index, axis=axis)


set_label = deprecated(
    version="0.2",
    reason=(
        "set_label has been renamed to assignlevel "
        "(and moved to dedicated index module)"
    ),
)(assignlevel)


def _projectlevel(index: Index, levels: Sequence[str]) -> Index:
    levels = np.atleast_1d(levels)
    if len(levels) == 1:
        return index.get_level_values(levels[0])

    return index.droplevel(index.names.difference(levels)).reorder_levels(levels)


def projectlevel(
    index_or_series: Union[Index, Series, DataFrame],
    levels: Sequence[str],
    axis: Union[int, str] = 0,
):
    """Project multiindex to given `levels`

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
    `MultiIndex.droplevel`, `Series.droplevel`, `DataFrame.droplevel`
    """
    if isinstance(index_or_series, Index):
        return _projectlevel(index_or_series, levels)

    index = index_or_series.index if axis in (0, "index") else index_or_series.columns
    return index_or_series.set_axis(_projectlevel(index, levels), axis=axis)


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


def ensure_multiindex(s):
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


def isin(df=None, **filters):
    """Constructs a MultiIndex selector

    Usage
    -----
    > df.loc[isin(region="World", gas=["CO2", "N2O"])]
    or with explicit df to get a boolean mask
    > isin(df, region="World", gas=["CO2", "N2O"])
    """

    def tester(df):
        if isinstance(df, Index):
            index = df
        else:
            index = df.index
        tests = (index.isin(np.atleast_1d(v), level=k) for k, v in filters.items())
        return reduce(and_, tests)

    return tester if df is None else tester(df)


def ismatch(df=None, singlefilter=None, regex=False, **filters):
    """Constructs an Index or MultiIndex selector based on pattern matching

    Usage
    -----
    > df.loc[ismatch(variable="Emissions|*|Fossil Fuel and Industry")]
    for a multiindex or
    > df.loc[ismatch("*bla*")]
    for a single index
    """

    def index_match(index, patterns):
        matches = np.zeros((len(index),), dtype=bool)
        for pat in patterns:
            if isinstance(pat, str):
                if not regex:
                    pat = shell_pattern_to_regex(pat) + "$"
                matches |= index.str.match(pat, na=False)
            else:
                matches |= index == pat
        return matches

    def multiindex_match(index, patterns, level):
        level_num = index.names.index(level)
        levels = index.levels[level_num]

        matches = index_match(levels, patterns)

        (indices,) = np.where(matches)
        return np.in1d(index.codes[level_num], indices)

    def tester(df):
        if isinstance(df, Index):
            index = df
        else:
            index = df.index

        if singlefilter is not None:
            matches = index_match(index, np.atleast_1d(singlefilter))
        else:
            tests = (
                multiindex_match(index, np.atleast_1d(v), level=k)
                for k, v in filters.items()
            )
            matches = reduce(and_, tests)

        return Series(matches, index=index)

    if df is None:
        return tester
    elif not isinstance(df, (DataFrame, Series)):
        # Special case: a pattern was passed in through `df` which wants to be applied to
        # hopefully a non-MultiIndex based Series or dataframe that we get afterwards
        singlefilter = df
        return tester
    else:
        return tester(df)
