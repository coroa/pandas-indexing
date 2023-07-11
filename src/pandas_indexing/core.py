"""
Core module.
"""
import re
from functools import reduce
from itertools import chain
from operator import and_, or_
from typing import (
    Any,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from deprecated import deprecated
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.core.indexes.frozen import FrozenList

from .types import Axis, Data, S, T
from .utils import doc, get_axis, print_list


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
    """Add or overwrite levels on a multiindex.

    Parameters
    ----------\
    {df}
    frame : Series or DataFrame, optional
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
    """Project multiindex to given ``levels``.

    Drops all levels except the ones explicitly mentioned from a given multiindex
    or an axis of a series or a dataframe.

    Parameters
    ----------\
    {index_or_data}
    levels : sequence of str
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

    index = get_axis(index_or_data, axis)
    return index_or_data.set_axis(_projectlevel(index, levels), axis=axis)


def concat(
    objs: Union[Iterable[S], Mapping[str, S]],
    order: Optional[Sequence[str]] = None,
    axis: Axis = 0,
    keys: Union[None, str, Index, Sequence] = None,
    copy: bool = False,
    **concat_kwds,
) -> S:
    """Concatenate pandas objects along a particular axis.

    In addition to the functionality provided by pd.concat, if the concat axis has a multiindex
    then the level order is reordered consistently.

    Parameters
    ----------
    objs : a sequence or mapping of Series or DataFrame objects
        If a mapping is passed the keys will be used as a new index level
        (with the name of the `keys` argument).
    order : a sequence of str, default None
        The order of level names in which to concatenate
    axis : Axis
        Axis along which to concatenate
    keys : str or list-like of str
        If `objs` is a mapping, a string-like value will be used as name of the new level,
        otherwise it is passed on to pd.concat.
    copy : bool, default False
        Whether to copy the underlying data
    **concat_kwds
        Other arguments accepted by pd.concat

    Returns
    -------
    Concatenated data

    Raises
    ------
    ValueError
        If the level names of `objs` do not match

    See also
    --------
    pandas.concat
    """

    if isinstance(objs, dict):
        objs = {k: v for k, v in objs.items() if v is not None}
        keys = Index(objs.keys(), name=keys)
        objs = list(objs.values())
    else:
        objs = [obj for obj in objs if obj is not None]

    if not objs:
        raise ValueError("Need at least one element to concatenate")

    if order is None:
        order = get_axis(objs[0], axis=axis).names
    elif isinstance(keys, Index):
        # make sure the order list does not include the new axis name
        order = [o for o in order if o != keys.name]

    orderset = frozenset(order)

    def reorder(df_or_ser):
        ax = get_axis(df_or_ser, axis=axis)
        if ax.names == order:
            return df_or_ser
        elif not set(ax.names) == orderset:
            raise ValueError(
                "All objects need to have the same index levels, but "
                f"{set(orderset)} != {set(ax.names)}"
            )

        return df_or_ser.set_axis(ax.reorder_levels(order), axis=axis, copy=False)

    return pd.concat(
        [reorder(o) for o in objs], keys=keys, copy=copy, axis=axis, **concat_kwds
    )


def _notna(
    index: Index,
    subset: Optional[Sequence[str]] = None,
    how: Literal["any", "all"] = "any",
) -> np.ndarray:
    if not isinstance(index, MultiIndex):
        return index.notna()

    subset = index.names if subset is None else np.atleast_1d(subset)
    codes = [index.codes[index.names.index(n)] for n in subset]
    op = and_ if how == "any" else or_
    return reduce(op, [c != -1 for c in codes])


def notna(
    index_or_data: Union[Index, Series, DataFrame],
    subset: Optional[Sequence[str]] = None,
    how: Literal["any", "all"] = "any",
    axis: Axis = 0,
):
    return _notna(get_axis(index_or_data, axis), subset, how)


def isna(
    index_or_data: Union[Index, Series, DataFrame],
    subset: Optional[Sequence[str]] = None,
    how: Literal["any", "all"] = "any",
    axis: Axis = 0,
):
    return ~_notna(get_axis(index_or_data, axis), subset, how)


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
    """Remove missing index values.

    Drops all index entries for which any or all (``how``) levels are
    undefined.

    Parameters
    ----------\
    {index_or_data}
    subset : Sequence[str], optional
        Names of levels on which to check for NA values
    how : {{"any", "all"}}
        Whether to remove an entry if all levels are NA or only a single one
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
    """Return unique index levels.

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
    """Describe index levels.

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
        if as_str is True

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


@doc(
    frame_or_series="""
    frame_or_series : DataFrame or Series
        Data to be filtered\
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
    """Semijoin ``data`` by index ``other``.

    Parameters
    ----------\
    {frame_or_series}
    other : Index
        Other index to join with
    how : {{'left', 'right', 'inner', 'outer'}}
        Join method to use
    level : None or str or int or
        Single level on which to join, if not given join on all
    sort : bool, optional
        Whether to sort the index
    axis : {{0, 1, "index", "columns"}}
        Axis on which to join

    Returns
    -------
    DataFrame or Series

    Raises
    ------
    ValueError
        If axis is not 0, "index" or 1, "columns"
    TypeError
        if frame_or_series does not derive from DataFrame or Series

    See also
    --------
    pandas.Index.join
    """

    index = get_axis(frame_or_series, axis)

    if level is None:
        index = ensure_multiindex(index)
        other = ensure_multiindex(other)

    new_index, left_idx, _ = index.join(
        other, how=how, level=level, return_indexers=True, sort=sort
    )

    if left_idx is None:
        return frame_or_series.set_axis(new_index, axis=axis)

    any_missing = not (left_idx != -1).all()
    if isinstance(frame_or_series, DataFrame):
        if axis in (0, "index"):
            data = frame_or_series.iloc[left_idx]
            index = data.index
        elif axis in (1, "columns"):
            data = frame_or_series.iloc[:, left_idx]
            index = data.columns
        if any_missing:
            data = data.where(pd.Series(left_idx != -1, index), axis=axis)
    elif isinstance(frame_or_series, Series):
        data = frame_or_series.iloc[left_idx]
        if any_missing:
            data = data.where(left_idx != -1)
    else:
        raise TypeError(
            f"frame_or_series must derive from DataFrame or Series, but is {type(frame_or_series)}"
        )

    return data.set_axis(new_index, axis=axis)


def _extractlevel(
    index: Index, template: Optional[str] = None, drop: bool = False, **templates: str
) -> Tuple[Index, List[str]]:
    index = ensure_multiindex(index)
    all_identifiers = set()

    if template is not None:
        if len(index.names) > 1:
            raise ValueError("``template`` may only be non-null for single index level")
        templates[index.names[0]] = template

    for dim, template in templates.items():
        identifiers = re.findall(r"\{([a-zA-Z_]+)\}", template)
        all_identifiers.update(identifiers)
        if dim not in index.names:
            raise ValueError(f"{dim} not a dimension of index: {index.names}")

        levelnum = index.names.index(dim)
        labels = index.levels[levelnum]
        codes = index.codes[levelnum]

        regex_pattern = reduce(
            lambda s, ident: s.replace(rf"\{{{ident}\}}", rf"(?P<{ident}>.*?)"),
            identifiers,
            re.escape(template),
        )
        components = labels.str.extract(f"^{regex_pattern}$", expand=True)

        index = assignlevel(
            index, **{ident: components[ident].values[codes] for ident in identifiers}
        )

    if drop:
        index = index.droplevel(list(set(templates) - all_identifiers))

    return index, list(all_identifiers)


@doc(
    index_or_data="""
    index_or_data : DataFrame, Series or Index
        Data to modify\
    """
)
def extractlevel(
    index_or_data: T,
    template: Optional[str] = None,
    *,
    drop: bool = False,
    dropna: bool = True,
    axis: Axis = 0,
    **templates: str,
) -> T:
    """Extract new index levels with ``templates`` matched against any index
    level.

    The ``**templates`` argument defines pairs of level names and templates.
    Given level names are matched against the template, f.ex. ``"Emi|{{gas}}|{{sector}}"``.
    Patterns (``{{gas}}`` or ``{{sector}}``) appearing in the template are extracted
    from the successful matches and added as new levels.

    Parameters
    ----------\
    {index_or_data}
    template : str, optional
        Extraction template for a single level
    drop : bool, default False
        Whether to keep the split dimension
    dropna : bool, default True
        Whether to drop the non-matching levels
    axis : {{0, 1, "index", "columns"}}, default 0
        Axis of DataFrame to extract from
    **templates : str
        Templates for splitting one or multiple levels

    Returns
    -------
    Index, Series or DataFrame

    Raises
    ------
    ValueError
        If ``dim`` is not a dimension of ``index_or_series``
    ValueError
        If ``template`` is given, while index has more than one level

    Examples
    --------
    >>> s = Series(
    ...     range(3),
    ...     MultiIndex.from_arrays(
    ...         [["SE|Elec|Bio", "SE|Elec|Coal", "PE|Coal"], ["GWh", "GWh", "EJ"]],
    ...         names=["variable", "unit"],
    ...     ),
    ... )
    >>> s
    variable      unit
    SE|Elec|Bio   GWh     0
    SE|Elec|Coal  GWh     1
    PE|Coal       EJ      2
    dtype: int64
    >>> extractlevel(s, variable="SE|{{type}}|{{fuel}}")
    variable      unit  type  fuel
    SE|Elec|Bio   GWh   Elec  Bio     0
    SE|Elec|Coal  GWh   Elec  Coal    1
    dtype: int64

    >>> extractlevel(s, variable="SE|{{type}}|{{fuel}}", dropna=False)
    variable      unit  type  fuel
    SE|Elec|Bio   GWh   Elec  Bio     0
    SE|Elec|Coal  GWh   Elec  Coal    1
    PE|Coal       EJ    NaN   NaN     2
    dtype: int64

    >>> s = Series(range(3), ["SE|Elec|Bio", "SE|Elec|Coal", "PE|Coal"])
    >>> extractlevel(s, "SE|{{type}}|{{fuel}}", drop=True)
    type  fuel
    Elec  Bio     0
          Coal    1
    dtype: int64

    See also
    --------
    formatlevel
    """
    if isinstance(index_or_data, Index):
        index_or_data, identifiers = _extractlevel(
            index_or_data, template, drop, **templates
        )
    else:
        index, identifiers = _extractlevel(
            get_axis(index_or_data, axis), template, drop, **templates
        )
        index_or_data = index_or_data.set_axis(index, axis=axis)

    if dropna:
        index_or_data = dropnalevel(index_or_data, subset=identifiers, axis=axis)

    return index_or_data


def _formatlevel(index: Index, drop: bool = False, **templates: str) -> Index:
    levels = {}
    used_levels = set()
    for dim, template in templates.items():
        # Build string
        string = ""
        prev_end = 0
        for m in re.finditer(r"\{([a-zA-Z_]+)\}", template):
            level = m.group(1)
            start, end = m.span()
            string += template[prev_end:start] + projectlevel(index, level).astype(str)
            prev_end = end
            used_levels.add(level)
        string += template[prev_end:]

        levels[dim] = string

    used_levels.difference_update(templates)

    if drop and used_levels:
        if len(used_levels) == len(index.levels):
            # none remain
            def expand_to_array(val):
                return (
                    val if not np.isscalar(val) else np.full(len(index), fill_value=val)
                )

            return MultiIndex.from_arrays(
                map(expand_to_array, levels.values()), names=levels.keys()
            )

        index = index.droplevel(list(used_levels))

    return assignlevel(index, **levels)


@doc(
    index_or_data="""
    index_or_data : DataFrame, Series or Index
        Data to modify\
    """
)
def formatlevel(
    index_or_data: T,
    drop: bool = False,
    axis: Axis = 0,
    **templates: str,
) -> T:
    """Format index levels based on a ``template`` which can refer to other
    levels.

    Parameters
    ----------\
    {index_or_data}
    drop : bool, default False
        Whether to drop the used index levels
    axis : {{0, 1, "index", "columns"}}, default 0
        Axis of DataFrame to modify
    **templates : str
        Format templates for one or multiple levels

    Returns
    -------
    Index, Series or DataFrame

    Raises
    ------
    ValueError
        If ``templates`` refer to non-existant levels
    """
    if isinstance(index_or_data, Index):
        return _formatlevel(index_or_data, drop, **templates)

    index = get_axis(index_or_data, axis)
    return index_or_data.set_axis(_formatlevel(index, drop, **templates), axis=axis)


def _fixindexna(index: Index):
    return index.set_codes(index.codes)


@doc(
    index_or_data="""
    index_or_data : Index, Series or DataFrame
        Datdropnalevel(self._obj, subset=subset, how=how, axis=axis)a
    """
)
def fixindexna(index_or_data: T, axis: Axis = 0) -> T:
    """Fix broken MultiIndex NA representation from .groupby(..., dropna=False)

    Refer to https://github.com/coroa/pandas-indexing/issues/25 for details

    Parameters
    ----------\
    {index_or_data}
    axis : Axis, optional
        Axis to fix, by default 0

    Returns
    -------
    index_or_data
    """
    if isinstance(index_or_data, Index):
        return _fixindexna(index_or_data)

    new_index = _fixindexna(get_axis(index_or_data, axis))
    return index_or_data.set_axis(new_index, axis=axis)
