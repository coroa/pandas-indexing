"""
Performs general tests.
"""

from textwrap import dedent

from numpy import nan
from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal, assert_index_equal

from pandas_indexing.core import (
    assignlevel,
    dropnalevel,
    projectlevel,
    summarylevel,
    uniquelevel,
)


def test_assignlevel_index(sidx: Index, midx: MultiIndex):
    """
    Checks scalar setting of new and old levels.
    """

    # Check adding to single index
    assert_index_equal(
        assignlevel(sidx, new=2),
        MultiIndex.from_arrays([sidx.values, [2, 2, 2]], names=["str", "new"]),
    )

    # Check whether adding multiple levels works
    assert_index_equal(
        assignlevel(sidx, new=2, new2=3),
        MultiIndex.from_arrays(
            [sidx.values, [2, 2, 2], [3, 3, 3]], names=["str", "new", "new2"]
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

    # Check adding from dataframe
    assert_index_equal(
        assignlevel(midx, DataFrame({"new": [1, 2, 3], "new2": 2})),
        MultiIndex.from_arrays(
            [midx.get_level_values(0), midx.get_level_values(1), [1, 2, 3], [2, 2, 2]],
            names=["str", "num", "new", "new2"],
        ),
    )

    # Check updating multilevel
    assert_index_equal(
        assignlevel(midx, num=2),
        MultiIndex.from_arrays(
            [midx.get_level_values(0), [2, 2, 2]], names=["str", "num"]
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


def test_assignlevel_dataframe(mdf: DataFrame):
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
                [mdf.columns.values, [2, 2]], names=[None, "new"]
            ),
        ),
    )


def test_dropnalevel(mdf):
    midx = MultiIndex.from_tuples(
        [("foo", nan, nan), ("foo", nan, 2), ("bar", 3, 3)],
        names=["str", "num", "num2"],
    )
    mdf = mdf.set_axis(midx)

    assert_index_equal(dropnalevel(midx), midx[[2]])
    assert_frame_equal(dropnalevel(mdf), mdf.iloc[[2]])

    assert_index_equal(
        dropnalevel(midx, subset=["num", "num2"], how="all"), midx[[1, 2]]
    )
    assert_index_equal(dropnalevel(midx, subset=["str", "num"]), midx[[2]])
    assert_index_equal(dropnalevel(midx, subset="num2"), midx[[1, 2]])


def test_projectlevel(midx, mdf):
    assert_frame_equal(
        projectlevel(mdf, "str"), mdf.set_axis(mdf.index.get_level_values("str"))
    )
    assert_index_equal(
        projectlevel(midx, ["num", "str"]),
        MultiIndex.from_arrays(
            [mdf.index.get_level_values("num"), mdf.index.get_level_values("str")]
        ),
    )


def test_uniquelevel(mdf, midx):
    assert_index_equal(uniquelevel(mdf, "str"), Index(["foo", "bar"], name="str"))

    assert_index_equal(uniquelevel(midx, ["str", "num"]), midx)


def test_summarylevel(mdf, midx):
    assert (
        summarylevel(mdf)
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

    assert (
        summarylevel(midx)
        == dedent(
            """
            Index:
             * str : foo, bar (2)
             * num : 1, 2, 3 (3)
            """
        ).strip()
    )
