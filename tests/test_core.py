"""
Performs general tests.
"""

from textwrap import dedent

import pytest
from numpy import nan, r_
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from pandas_indexing.core import (
    assignlevel,
    concat,
    describelevel,
    dropnalevel,
    extractlevel,
    formatlevel,
    projectlevel,
    semijoin,
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


def test_assignlevel_data(mdf: DataFrame):
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


def test_formatlevel_options(mdf: DataFrame):
    idx_str = mdf.index.get_level_values(0)
    idx_num = mdf.index.get_level_values(1)

    # drop
    assert_frame_equal(
        formatlevel(mdf, new="{str}|{num}", drop=True),
        mdf.set_axis(
            MultiIndex.from_arrays([idx_str + "|" + idx_num.astype(str)], names=["new"])
        ),
    )

    # axis=1
    mdf_t = mdf.T
    assert_frame_equal(
        formatlevel(mdf_t, new="{str}|{num}", axis=1),
        mdf_t.set_axis(
            MultiIndex.from_arrays(
                [idx_str, idx_num, idx_str + "|" + idx_num.astype(str)],
                names=["str", "num", "new"],
            ),
            axis=1,
        ),
    )


def test_formatlevel_data(mdf, mseries, midx):
    idx_str = midx.get_level_values(0)
    idx_num = midx.get_level_values(1)
    expected_idx = MultiIndex.from_arrays(
        [
            idx_str,
            idx_num,
            idx_str + "|" + idx_num.astype(str),
            idx_num.astype(str) + " 2",
        ],
        names=["str", "num", "new", "new2"],
    )
    assert_frame_equal(
        formatlevel(mdf, new="{str}|{num}", new2="{num} 2"), mdf.set_axis(expected_idx)
    )
    assert_series_equal(
        formatlevel(mseries, new="{str}|{num}", new2="{num} 2"),
        mseries.set_axis(expected_idx),
    )
    assert_index_equal(
        formatlevel(midx, new="{str}|{num}", new2="{num} 2"), expected_idx
    )


def test_extractlevel(mdf, mseries, midx):
    midx = MultiIndex.from_arrays(
        [["e|foo", "e|bar", "bar"], [1, 2, 3]], names=["var", "num"]
    )
    mdf = mdf.set_axis(midx)
    mseries = mseries.set_axis(midx)

    expected_idx = MultiIndex.from_arrays(
        [["e|foo", "e|bar"], [1, 2], ["e", "e"], ["foo", "bar"]],
        names=["var", "num", "e", "typ"],
    )

    assert_index_equal(extractlevel(midx, var="{e}|{typ}"), expected_idx)

    assert_series_equal(
        extractlevel(mseries, var="{e}|{typ}"),
        mseries.iloc[[0, 1]].set_axis(expected_idx),
    )

    assert_frame_equal(
        extractlevel(mdf, var="{e}|{typ}"), mdf.iloc[[0, 1]].set_axis(expected_idx)
    )


def test_extractlevel_options(mdf):
    midx = MultiIndex.from_arrays(
        [["e|foo", "e|bar", "bar"], [1, 2, 3]], names=["var", "num"]
    )
    mdf_t = mdf.T.set_axis(midx, axis=1)

    # drop=True
    assert_index_equal(
        extractlevel(midx, var="{e}|{typ}", drop=True),
        MultiIndex.from_arrays(
            [[1, 2], ["e", "e"], ["foo", "bar"]],
            names=["num", "e", "typ"],
        ),
    )

    # dropna=False
    assert_index_equal(
        extractlevel(midx, var="{e}|{typ}", dropna=False),
        MultiIndex.from_arrays(
            [
                ["e|foo", "e|bar", "bar"],
                [1, 2, 3],
                ["e", "e", nan],
                ["foo", "bar", nan],
            ],
            names=["var", "num", "e", "typ"],
        ),
    )

    # axis=1
    assert_frame_equal(
        extractlevel(mdf_t, var="{e}|{typ}", drop=True, axis=1),
        mdf_t.iloc[:, [0, 1]].set_axis(
            MultiIndex.from_arrays(
                [[1, 2], ["e", "e"], ["foo", "bar"]],
                names=["num", "e", "typ"],
            ),
            axis=1,
        ),
    )

    with pytest.raises(ValueError):
        # mdf does not have the var level
        extractlevel(mdf, var="{e}|{typ}")


def test_extractlevel_single(midx):
    sidx = Index(["e|foo", "e|bar", "bar"])
    assert_index_equal(
        extractlevel(sidx, "{e}|{typ}", drop=True),
        MultiIndex.from_arrays([["e", "e"], ["foo", "bar"]], names=["e", "typ"]),
    )

    sidx = Index(["e|foo", "e|bar", "bar"], name="named")
    assert_index_equal(
        extractlevel(sidx, "{e}|{typ}", drop=True),
        MultiIndex.from_arrays([["e", "e"], ["foo", "bar"]], names=["e", "typ"]),
    )

    with pytest.raises(ValueError):
        # MultiIndex input with single level template
        extractlevel(midx, "{e}|{typ}")


def test_concat(mdf):
    assert_frame_equal(concat([mdf.iloc[:1], mdf.iloc[1:]]), mdf)

    assert_frame_equal(concat([mdf.iloc[:1], None, mdf.iloc[1:].swaplevel()]), mdf)

    assert_frame_equal(
        concat(
            {"part1": mdf.iloc[:1], "part2": mdf.iloc[1:].swaplevel(), "skip": None},
            keys="new",
        ),
        mdf.set_axis(
            MultiIndex.from_arrays(
                [
                    ["part1", "part2", "part2"],
                    mdf.index.get_level_values(0),
                    mdf.index.get_level_values(1),
                ],
                names=["new", "str", "num"],
            )
        ),
    )

    assert_frame_equal(concat([mdf.iloc[:, :1], mdf.iloc[:, 1:]], axis=1), mdf)

    with pytest.raises(ValueError):
        concat([mdf.iloc[:1], mdf.iloc[1:].droplevel("num")])

    with pytest.raises(ValueError):
        concat([])


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


def test_describelevel(mdf, midx):
    assert (
        describelevel(mdf, as_str=True)
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
        describelevel(midx, as_str=True)
        == dedent(
            """
            Index:
             * str : foo, bar (2)
             * num : 1, 2, 3 (3)
            """
        ).strip()
    )


def test_semijoin(mdf, mseries):
    index = MultiIndex.from_tuples(
        [(2.5, "foo", 2), (3.5, "bar", 3), (5.5, "bar", 5)], names=["new", "str", "num"]
    )

    # Left-join
    assert_frame_equal(
        semijoin(mdf, index, how="left"),
        DataFrame(
            {col: mdf[col].values for col in mdf},
            index=assignlevel(mdf.index, new=[nan, 2.5, 3.5]),
        ),
    )

    # Inner-join
    assert_frame_equal(
        semijoin(mdf, index, how="inner"),
        DataFrame(
            {col: mdf[col].values[1:3] for col in mdf},
            index=assignlevel(mdf.index[1:3], new=[2.5, 3.5]),
        ),
    )

    # Right-join
    assert_frame_equal(
        semijoin(mdf, index, how="right"),
        DataFrame(
            {col: r_[mdf[col].values[1:3], nan] for col in mdf},
            index=index.reorder_levels(["str", "num", "new"]),
        ),
    )

    # Right-join on series
    assert_series_equal(
        semijoin(mseries, index, how="right"),
        Series(
            r_[mseries.values[1:3], nan],
            index=index.reorder_levels(["str", "num", "new"]),
        ),
    )
