import pytest
from numpy import array
from numpy.testing import assert_array_equal
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal

from pandas_indexing import isin, ismatch
from pandas_indexing.selectors import (
    AllSelector,
    And,
    Const,
    Isin,
    Ismatch,
    NoneSelector,
    Not,
    Or,
)
from pandas_indexing.utils import EqualIdentity


def test_isin_mseries(mseries: Series):
    sel = isin(str="foo")
    assert isinstance(sel, Isin)
    assert_array_equal(sel(mseries), array([True, True, False]))
    assert_series_equal(mseries.loc[sel], Series([1, 2], mseries.index[:2]))

    sel = isin(num=[2, 3], str="foo")
    assert_series_equal(mseries.loc[sel], mseries.iloc[[1]])

    sel = isin(num=lambda s: s > 1)
    assert_series_equal(mseries.loc[sel], mseries.iloc[[1, 2]])


def test_isin_ignore_missing_levels(mseries: Series):
    sel = isin(str="foo", bla=False)
    with pytest.raises(KeyError):
        mseries.loc[sel]

    sel = isin(str="foo", bla=False, ignore_missing_levels=True)
    assert_series_equal(mseries.loc[sel], Series([1, 2], mseries.index[:2]))


def test_isin_operations(mdf: DataFrame):
    sel = isin(str="foo") & ~isin(num=2)
    assert sel == And(Isin(dict(str="foo")), Not(Isin(dict(num=2))))

    assert_frame_equal(mdf.loc[sel], mdf.iloc[[0]])

    s_b = Series([False, True, False], mdf.index)
    sel = isin(str="bar") | s_b
    assert sel == Or(Isin(dict(str="bar")), Const(EqualIdentity(s_b)))

    sel = s_b & isin(str="bar")
    assert sel == And(Isin(dict(str="bar")), Const(EqualIdentity(s_b)))


def test_ismatch_single(sdf: DataFrame):
    sel = ismatch("b*")
    assert isinstance(sel, Ismatch)
    assert_frame_equal(sdf.loc[sel], sdf.iloc[[1, 2]])


def test_ismatch_explicitly_given(sdf):
    assert_series_equal(ismatch(sdf, "b*"), Series([False, True, True], sdf.index))

    assert_series_equal(
        ismatch(sdf.columns, ["o*"]), Series([True, False], sdf.columns)
    )


def test_all_none(sdf):
    assert isinstance(~AllSelector(), NoneSelector)
    assert isinstance(~NoneSelector(), AllSelector)

    sel = isin(str="bar")

    assert isinstance(sel | AllSelector(), AllSelector)
    assert isinstance(AllSelector() | sel, AllSelector)
    assert sel & AllSelector() == sel
    assert AllSelector() & sel == sel

    assert sel | NoneSelector() == sel
    assert NoneSelector() | sel == sel
    assert isinstance(sel & NoneSelector(), NoneSelector)
    assert isinstance(NoneSelector() & sel, NoneSelector)

    assert_frame_equal(sdf.loc[AllSelector()], sdf)
    assert_frame_equal(sdf.loc[NoneSelector()], sdf.iloc[[]])
