"""
Performs general tests.
"""

from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal, assert_index_equal

from pandas_indexing.core import assignlevel


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
