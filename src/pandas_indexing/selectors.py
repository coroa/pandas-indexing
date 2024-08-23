"""
Selectors improve ``.loc[]`` indexing for multi-index pandas data.
"""

from functools import reduce
from operator import and_
from typing import Any, Mapping, Optional, Union

import numpy as np
from attrs import define, field
from pandas import DataFrame, Index, Series

from .types import Data
from .utils import shell_pattern_to_regex


def maybe_const(x):
    return x if isinstance(x, Selector) else Const(x)


class Selector:
    # Tell numpy and pandas that we want precedence
    __array_ufunc__ = None
    __pandas_priority__ = 5000

    def __invert__(self):
        return Not(self)

    def __and__(self, other):
        if isinstance(other, Special):
            return other & self
        return And(self, maybe_const(other))

    def __or__(self, other):
        if isinstance(other, Special):
            return other | self
        return Or(self, maybe_const(other))

    __rand__ = __and__
    __ror__ = __or__


@define
class BinOp(Selector):
    a: Selector
    b: Selector


@define
class Const(Selector):
    val: Any

    def __call__(self, df):
        return self.val(df) if callable(self.val) else self.val


@define
class And(BinOp):
    def __call__(self, df):
        return self.a(df) & self.b(df)


@define
class Or(BinOp):
    def __call__(self, df):
        return self.a(df) | self.b(df)


@define
class Not(Selector):
    a: Selector

    def __call__(self, df):
        return ~self.a(df)


class Special(Selector):
    pass


@define
class AllSelector(Special):
    def __invert__(self):
        return NoneSelector()

    def __and__(self, other):
        return other

    def __or__(self, other):
        return self

    def __call__(self, df):
        return Series(True, df.index)


@define
class NoneSelector(Special):
    def __invert__(self):
        return AllSelector()

    def __and__(self, other):
        return self

    def __or__(self, other):
        return other

    __rand__ = __and__
    __ror__ = __or__

    def __call__(self, df):
        return Series(False, df.index)


# Singletons for easy access
All = AllSelector()
None_ = NoneSelector()


@define
class Isin(Selector):
    filters: Mapping[str, Any]
    ignore_missing_levels: bool = field(default=False, repr=False)

    def __call__(self, df):
        if isinstance(df, Index):
            index = df
        else:
            index = df.index

        filters = self.filters
        if self.ignore_missing_levels:
            filters = {k: v for k, v in filters.items() if k in index.names}

        def apply_filter(value, level):
            if callable(value):
                return value(index.get_level_values(level))
            return index.isin(np.atleast_1d(value), level=level)

        return Series(
            reduce(and_, (apply_filter(v, k) for k, v in filters.items())), index
        )


def isin(
    df: Optional[Data] = None, ignore_missing_levels: bool = False, **filters: Any
) -> Union[Isin, Series]:
    """Constructs a MultiIndex selector.

    Arguments
    ---------
    df : Data, optional
        Data on which to match, if missing an ``Isin`` object is returned
    ignore_missing_levels : bool, default False
        If set, levels missing in data index will be ignored
    **filters
        Filter to apply on given levels (lists are ``or`` ed, levels are ``and`` ed)
        Callables are evaluated on the index level values.

    Returns
    -------
    Isin or Series

    Example
    -------
    >>> df.loc[isin(region="World", gas=["CO2", "N2O"])]

    or with explicit df to get a boolean mask

    >>> isin(df, region="World", gas=["CO2", "N2O"])
    """

    tester = Isin(filters, ignore_missing_levels=ignore_missing_levels)
    return tester if df is None else tester(df)


@define
class Ismatch(Selector):
    filters: Mapping[str, Any]
    regex: bool = False
    ignore_missing_levels: bool = field(default=False, repr=False)

    def index_match(self, index, patterns):
        matches = np.zeros((len(index),), dtype=bool)
        for pat in patterns:
            if isinstance(pat, str):
                if not self.regex:
                    pat = shell_pattern_to_regex(pat) + "$"
                matches |= index.str.match(pat, na=False)
            else:
                matches |= index == pat
        return matches

    def multiindex_match(self, index, patterns, level):
        level_num = index.names.index(level)
        levels = index.levels[level_num]

        matches = self.index_match(levels, patterns)

        (indices,) = np.where(matches)
        return np.in1d(index.codes[level_num], indices)

    def __call__(self, df):
        if isinstance(df, Index):
            index = df
        else:
            index = df.index

        single = self.filters.get(None) if len(self.filters) == 1 else None
        if single is not None:
            matches = self.index_match(index, np.atleast_1d(single))
        else:
            filters = self.filters
            if self.ignore_missing_levels:
                filters = {k: v for k, v in filters.items() if k in index.names}

            tests = (
                self.multiindex_match(index, np.atleast_1d(v), level=k)
                for k, v in filters.items()
            )
            matches = reduce(and_, tests)

        return Series(matches, index=index)


def ismatch(
    df: Union[None, Index, DataFrame, Series, str] = None,
    singlefilter: Optional[str] = None,
    regex: bool = False,
    ignore_missing_levels: bool = False,
    **filters,
) -> Union[Ismatch, Series]:
    """Constructs an Index or MultiIndex selector based on pattern matching.

    Arguments
    ---------
    df : Data, optional
        Data on which to match, if missing an ``Isin`` object is returned.
    singlefilter : str, optional
        Filter to apply on a non-multiindex index (can also be handed into the ``df``
        argument)
    regex : bool, default False
        If set, filters are interpreted as plain regex strings, otherwise (by default) a
        glob-like syntax is used
    ignore_missing_levels : bool, default False
        If set, levels missing in data index will be ignored
    **filters
        Filter to apply on given levels (lists are ``or`` ed, levels are ``and`` ed)

    Returns
    -------
    Isin or Series

    Example
    -------
    for a multiindex:

    >>> df.loc[ismatch(variable="Emissions|*|Fossil Fuel and Industry")]

    for a single index:

    >>> df.loc[ismatch("*bla*")]
    """

    if not filters and singlefilter is not None:
        filters = {None: singlefilter}

    if df is None:
        return Ismatch(
            filters, regex=regex, ignore_missing_levels=ignore_missing_levels
        )
    elif not isinstance(df, (DataFrame, Series, Index)):
        # Special case: a pattern was passed in through `df` which wants to be applied to
        # hopefully a non-MultiIndex based Series or dataframe that we get afterwards
        filters = {None: df}
        return Ismatch(filters, regex=regex)
    else:
        return Ismatch(
            filters, regex=regex, ignore_missing_levels=ignore_missing_levels
        )(df)
