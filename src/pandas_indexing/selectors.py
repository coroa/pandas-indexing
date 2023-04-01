"""
Selectors improve .loc[] indexing for multi-index pandas data.
"""

from functools import reduce
from operator import and_
from typing import Any

import numpy as np
from attrs import define
from pandas import DataFrame, Index, Series

from .utils import shell_pattern_to_regex


def maybe_const(x):
    return x if isinstance(x, Selector) else Const(x)


class Selector:
    # Tell numpy that we want precedence
    __array_ufunc__ = None

    def reshape(self, _):
        return self

    def __invert__(self):
        return Not(self)

    def __and__(self, other):
        return And(self, maybe_const(other))

    def __or__(self, other):
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

    def __call__(self, _):
        return self.val


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
        return ~self.a.__call__(df)


@define
class Isin(Selector):
    filters: dict[str, Any]

    def __call__(self, df):
        if isinstance(df, Index):
            index = df
        else:
            index = df.index
        tests = (index.isin(np.atleast_1d(v), level=k) for k, v in self.filters.items())
        return reduce(and_, tests)


def isin(df=None, **filters):
    """
    Constructs a MultiIndex selector.

    Usage
    -----
    >>> df.loc[isin(region="World", gas=["CO2", "N2O"])]

    or with explicit df to get a boolean mask

    >>> isin(df, region="World", gas=["CO2", "N2O"])
    """

    tester = Isin(filters)
    return tester if df is None else tester(df)


@define
class Ismatch(Selector):
    filters: dict[str, Any]
    regex: bool = False

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
            tests = (
                self.multiindex_match(index, np.atleast_1d(v), level=k)
                for k, v in self.filters.items()
            )
            matches = reduce(and_, tests)

        return Series(matches, index=index)


def ismatch(df=None, singlefilter=None, regex=False, **filters):
    """
    Constructs an Index or MultiIndex selector based on pattern matching.

    Usage
    -----
    for a multiindex:

    >>> df.loc[ismatch(variable="Emissions|*|Fossil Fuel and Industry")]

    for a single index:

    >>> df.loc[ismatch("*bla*")]
    """

    if not filters and singlefilter is not None:
        filters = {None: singlefilter}

    if df is None:
        return Ismatch(filters, regex)
    elif not isinstance(df, (DataFrame, Series)):
        # Special case: a pattern was passed in through `df` which wants to be applied to
        # hopefully a non-MultiIndex based Series or dataframe that we get afterwards
        filters = {None: df}
        return Ismatch(filters, regex)
    else:
        return Ismatch(filters, regex)(df)
