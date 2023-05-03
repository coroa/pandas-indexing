"""
Performs general tests.
"""

from textwrap import dedent

from numpy import nan
from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal, assert_index_equal

import pandas_indexing.accessors  # noqa: F401


def test_assign_index(midx: MultiIndex):
    """
    Checks scalar setting of new and old levels.
    """

    # Check for adding to multilevel
    assert_index_equal(
        midx.idx.assign(new=[1, 2, 3]),
        MultiIndex.from_arrays(
            [midx.get_level_values(0), midx.get_level_values(1), [1, 2, 3]],
            names=["str", "num", "new"],
        ),
    )

    # Check adding from dataframe
    assert_index_equal(
        midx.idx.assign(DataFrame({"new": [1, 2, 3], "new2": 2})),
        MultiIndex.from_arrays(
            [midx.get_level_values(0), midx.get_level_values(1), [1, 2, 3], [2, 2, 2]],
            names=["str", "num", "new", "new2"],
        ),
    )


def test_assign_dataframe(mdf: DataFrame):
    """
    Checks setting dataframes on both axes.
    """
    assert_frame_equal(
        mdf.idx.assign(new=2),
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
        mdf.idx.assign(new=2, axis=1),
        DataFrame(
            mdf.values,
            index=mdf.index,
            columns=MultiIndex.from_arrays(
                [mdf.columns.values, [2, 2]], names=[None, "new"]
            ),
        ),
    )


def test_dropna(mdf):
    midx = MultiIndex.from_tuples(
        [("foo", nan, nan), ("foo", nan, 2), ("bar", 3, 3)],
        names=["str", "num", "num2"],
    )
    mdf = mdf.set_axis(midx)

    assert_index_equal(midx.idx.dropna(subset=["num", "num2"], how="all"), midx[[1, 2]])


def test_project(midx, mdf):
    assert_frame_equal(
        mdf.idx.project("str"), mdf.set_axis(mdf.index.get_level_values("str"))
    )
    assert_index_equal(
        midx.idx.project(["num", "str"]),
        MultiIndex.from_arrays(
            [mdf.index.get_level_values("num"), mdf.index.get_level_values("str")]
        ),
    )


def test_repr(mdf):
    assert (
        str(mdf.idx)
        == dedent(
            """
            Index:
             * str : foo, bar (2)
             * num : 1, 2, 3 (3)

            Columns:
             * <unnamed> : one, two (2)
            """
        ).strip()
    )


def test_unique(mdf, midx):
    assert_index_equal(mdf.idx.unique("str"), Index(["foo", "bar"], name="str"))
