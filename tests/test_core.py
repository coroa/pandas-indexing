"""
Performs general tests.
"""

import numpy as np
import pytest
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.testing import assert_frame_equal, assert_index_equal

from pandas_indexing.core import assignlevel


@pytest.fixture
def midx() -> MultiIndex:
    return MultiIndex.from_tuples(
        [("foo", 1), ("foo", 2), ("bar", 3)], names=["str", "num"]
    )


@pytest.fixture
def sidx() -> Index:
    return Index(["foo", "bar", "baz"], name="str")


@pytest.fixture
def mseries(midx) -> Series:
    return Series([1, 2, 3], midx)


@pytest.fixture
def sseries(sidx) -> Series:
    return Series([1, 2, 3], sidx)


@pytest.fixture
def sdf(sidx) -> DataFrame:
    return DataFrame(dict(one=1, two=[1, 2, 3]), sidx)


@pytest.fixture
def mdf(midx) -> DataFrame:
    return DataFrame(dict(one=1, two=[1, 2, 3]), midx)


def test_assignlevel_index(sidx, midx):
    """
    Checks scalar setting of new and old levels.
    """

    # Check adding to single index
    assert_index_equal(
        assignlevel(sidx, new=2),
        MultiIndex.from_arrays([sidx.values, np.full(3, 2)], names=["str", "new"]),
    )

    # Check whether adding multiple levels works
    assert_index_equal(
        assignlevel(sidx, new=2, new2=3),
        MultiIndex.from_arrays(
            [sidx.values, np.full(3, 2), np.full(3, 3)], names=["str", "new", "new2"]
        ),
    )

    # Check for adding to multilevel
    assert_index_equal(
        assignlevel(midx, new=[1, 2, 3]),
        MultiIndex.from_arrays(
            [midx.get_level_values(0), midx.get_level_values(1), [1, 2, 3]],
            names=["str", "num", "new"],
        ),
    )

    # Check updating multilevel
    assert_index_equal(
        assignlevel(midx, num=2),
        MultiIndex.from_arrays(
            [midx.get_level_values(0), np.full(3, 2)],
            names=["str", "num"],
        ),
    )

    # Check order
    assert_index_equal(
        assignlevel(midx, new=[1, 2, 3], order=["num", "new", "str"]),
        MultiIndex.from_arrays(
            [midx.get_level_values(1), [1, 2, 3], midx.get_level_values(0)],
            names=["num", "new", "str"],
        ),
    )


def test_assignlevel_dataframe(mdf):
    """
    Checks setting dataframes on both axes.
    """
    assert_frame_equal(
        assignlevel(mdf, new=2),
        DataFrame(
            mdf.values,
            index=MultiIndex.from_arrays(
                [
                    mdf.index.get_level_values(0),
                    mdf.index.get_level_values(1),
                    [2, 2, 2],
                ],
                names=["str", "num", "new"],
            ),
            columns=mdf.columns,
        ),
    )

    assert_frame_equal(
        assignlevel(mdf, new=2, axis=1),
        DataFrame(
            mdf.values,
            index=mdf.index,
            columns=MultiIndex.from_arrays(
                [mdf.columns.values, np.full(2, 2)],
                names=[None, "new"],
            ),
        ),
    )
