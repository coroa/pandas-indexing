"""
Performs general tests.
"""

import sys
from re import escape
from textwrap import dedent

import pytest
from numpy import array, nan, r_
from numpy.testing import assert_array_equal
from pandas import DataFrame, Index, MultiIndex, Series
from pandas import concat as pd_concat
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from pandas_indexing.core import (
    add_zeros_like,
    aggregatelevel,
    antijoin,
    assignlevel,
    concat,
    describelevel,
    dropnalevel,
    extractlevel,
    formatlevel,
    isna,
    notna,
    projectlevel,
    semijoin,
    to_tidy,
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

    # Check adding from series (auto-alignment will switch first and second line)
    assert_index_equal(
        assignlevel(
            midx, Series([1, 2, 3], name="new", index=Index([2, 1, 3], name="num"))
        ),
        MultiIndex.from_arrays(
            [midx.get_level_values(0), midx.get_level_values(1), [2, 1, 3]],
            names=["str", "num", "new"],
        ),
    )

    # Check adding from dataframe (auto-alignment will switch first and second line)
    assert_index_equal(
        assignlevel(
            midx,
            DataFrame(
                {"new": [1, 2, 3], "new2": 2}, index=Index([2, 1, 3], name="num")
            ),
        ),
        MultiIndex.from_arrays(
            [midx.get_level_values(0), midx.get_level_values(1), [2, 1, 3], [2, 2, 2]],
            names=["str", "num", "new", "new2"],
        ),
    )

    # Check adding from dataframe w/o auto-alignment
    assert_index_equal(
        assignlevel(
            midx,
            DataFrame(
                {"new": [1, 2, 3], "new2": 2}, index=Index([2, 1, 3], name="num")
            ),
            ignore_index=True,
        ),
        MultiIndex.from_arrays(
            [midx.get_level_values(0), midx.get_level_values(1), [1, 2, 3], [2, 2, 2]],
            names=["str", "num", "new", "new2"],
        ),
    )

    # Check wrong types
    with pytest.raises(ValueError):
        assignlevel(midx, "no-frame")

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

    with pytest.raises(ValueError):
        assignlevel(mdf, new=2, axis="no-axis")

    with pytest.raises(ValueError):
        assignlevel("no-data", new=2)


def test_formatlevel_options(mdf: DataFrame):
    idx_str = mdf.index.get_level_values(0)
    idx_num = mdf.index.get_level_values(1)

    # drop
    assert_frame_equal(
        formatlevel(mdf, new="{str}|{num}", drop=True),
        mdf.set_axis(Index(idx_str + "|" + idx_num.astype(str), name="new")),
    )

    assert_frame_equal(
        formatlevel(mdf, new="{str}|{num}", str="{str}|other", drop=True),
        mdf.set_axis(
            MultiIndex.from_arrays(
                [idx_str + "|other", idx_str + "|" + idx_num.astype(str)],
                names=["str", "new"],
            )
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
        [[1, 2], ["e", "e"], ["foo", "bar"]],
        names=["num", "e", "typ"],
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

    # keep=True
    assert_index_equal(
        extractlevel(midx, var="{e}|{typ}", keep=True),
        MultiIndex.from_arrays(
            [["e|foo", "e|bar"], [1, 2], ["e", "e"], ["foo", "bar"]],
            names=["var", "num", "e", "typ"],
        ),
    )

    # dropna=False
    assert_index_equal(
        extractlevel(midx, var="{e}|{typ}", dropna=False),
        MultiIndex.from_arrays(
            [[1, 2, 3], ["e", "e", nan], ["foo", "bar", nan]],
            names=["num", "e", "typ"],
        ),
    )

    # axis=1
    assert_frame_equal(
        extractlevel(mdf_t, var="{e}|{typ}", axis=1),
        mdf_t.iloc[:, [0, 1]].set_axis(
            MultiIndex.from_arrays(
                [[1, 2], ["e", "e"], ["foo", "bar"]],
                names=["num", "e", "typ"],
            ),
            axis=1,
        ),
    )

    # regex
    assert_index_equal(
        extractlevel(midx, var=r"((?P<e>.*?)\|)?(?P<typ>.*?)", regex=True),
        MultiIndex.from_arrays(
            [[1, 2, 3], ["e", "e", nan], ["foo", "bar", "bar"]],
            names=["num", "e", "typ"],
        ),
    )

    # drop=True
    with pytest.warns(DeprecationWarning):
        assert_index_equal(
            extractlevel(midx, var="{e}|{typ}", drop=False),
            MultiIndex.from_arrays(
                [["e|foo", "e|bar"], [1, 2], ["e", "e"], ["foo", "bar"]],
                names=["var", "num", "e", "typ"],
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


def test_concat(mdf, midx, sidx):
    assert_frame_equal(concat([mdf.iloc[:1], mdf.iloc[1:]]), mdf)

    assert_frame_equal(
        concat(
            [mdf.iloc[:1], mdf.iloc[1:]],
            keys=Index([1, 2], name="new"),
            order=["num", "str", "new"],
        ),
        pd_concat(
            [mdf.iloc[:1], mdf.iloc[1:]], keys=Index([1, 2], name="new")
        ).reorder_levels(["new", "num", "str"]),
    )

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

    def maybe_swap(idx):
        return idx.swaplevel() if isinstance(idx, MultiIndex) else idx

    for idx in (midx, sidx):
        assert_index_equal(concat([idx[:1], idx[1:]]), idx)

        assert_index_equal(concat([idx[:1], None, maybe_swap(idx[1:])]), idx)

        assert_index_equal(
            concat(
                {"part1": idx[:1], "part2": maybe_swap(idx[1:]), "skip": None},
                keys="new",
            ),
            assignlevel(idx, new=["part1", "part2", "part2"]),
        )

    with pytest.raises(ValueError):
        concat([mdf.iloc[:1], mdf.iloc[1:].droplevel("num")])

    with pytest.raises(ValueError):
        concat([])


def test_isna(mdf, mseries):
    midx = MultiIndex.from_tuples(
        [("foo", nan, nan), ("foo", nan, 2), ("bar", 3, 3)],
        names=["str", "num", "num2"],
    )
    for x in [mdf.set_axis(midx), mseries.set_axis(midx), midx]:
        assert_array_equal(isna(x), array([True, True, False]))
        assert_array_equal(notna(x), array([False, False, True]))

    assert_array_equal(isna(x, subset=["num"]), array([True, True, False]))
    assert_array_equal(notna(x, subset=["num"]), array([False, False, True]))

    assert_array_equal(isna(x, how="all"), array([False, False, False]))
    assert_array_equal(
        isna(x, subset=["num", "num2"], how="all"), array([True, False, False])
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
    with pytest.raises(
        KeyError,
        match=escape('Index has no level "blub" (existing levels: "str", "num")'),
    ):
        projectlevel(mdf, "blub")


def test_uniquelevel(mdf, midx):
    assert_index_equal(uniquelevel(mdf, "str"), Index(["foo", "bar"], name="str"))

    assert_index_equal(uniquelevel(midx, ["str", "num"]), midx)


def test_describelevel(mdf, mseries, midx):
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
        describelevel(mseries, as_str=True)
        == dedent(
            """
            Index:
             * str : foo, bar (2)
             * num : 1, 2, 3 (3)
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

    # Python 3.12 changes the level order
    level_order = (
        ["new", "str", "num"]
        if sys.version_info >= (3, 12, 0)
        else ["str", "num", "new"]
    )

    # Right-join
    assert_frame_equal(
        semijoin(mdf, index, how="right"),
        DataFrame(
            {col: r_[mdf[col].values[1:3], nan] for col in mdf},
            index=index.reorder_levels(level_order),
        ),
    )

    # Right-join on series
    assert_series_equal(
        semijoin(mseries, index, how="right"),
        Series(r_[mseries.values[1:3], nan], index=index.reorder_levels(level_order)),
    )


def test_antijoin(mdf, mseries):
    index = MultiIndex.from_tuples(
        [(2.5, "foo", 2), (3.5, "bar", 3), (5.5, "bar", 5)], names=["new", "str", "num"]
    )

    # Frame
    assert_frame_equal(antijoin(mdf, index), mdf.iloc[[0]])

    # Series
    assert_series_equal(antijoin(mseries, index), mseries.iloc[[0]])

    # Index
    assert_index_equal(antijoin(mseries.index, index), mseries.index[[0]])


def test_to_tidy(mdf, mseries, midx):
    assert_frame_equal(
        to_tidy(mdf),
        DataFrame(
            dict(
                str=["foo", "foo", "foo", "foo", "bar", "bar"],
                num=[1, 1, 2, 2, 3, 3],
                year=["one", "two"] * 3,
                value=[1, 1, 1, 2, 1, 3],
            )
        ),
    )

    assert_frame_equal(
        to_tidy(mseries),
        DataFrame(dict(str=["foo", "foo", "bar"], num=[1, 2, 3], value=[1, 2, 3])),
    )

    with pytest.raises(TypeError):
        to_tidy(midx)


@pytest.mark.parametrize(
    "value_name, columns", [[None, None], ["value_name", "year_name"]]
)
def test_to_tidy_names(mdf, value_name, columns):
    mdf = mdf.rename_axis(columns="columns")

    assert_frame_equal(
        to_tidy(mdf, value_name=value_name, columns=columns),
        DataFrame(
            {
                "str": ["foo", "foo", "foo", "foo", "bar", "bar"],
                "num": [1, 1, 2, 2, 3, 3],
                columns or "columns": ["one", "two"] * 3,
                value_name or 0: [1, 1, 1, 2, 1, 3],
            }
        ),
    )


def test_aggregatelevel(mdf):
    # replace
    assert_frame_equal(
        aggregatelevel(mdf, num=dict(new=[1, 2])),
        DataFrame(
            dict(one=[1, 2], two=[3, 3]),
            MultiIndex.from_tuples([("bar", 3), ("foo", "new")], names=["str", "num"]),
        ),
    )

    # text axis
    assert_frame_equal(
        aggregatelevel(mdf.T, num=dict(new=[1, 2]), axis=1),
        DataFrame(
            dict(one=[1, 2], two=[3, 3]),
            MultiIndex.from_tuples([("bar", 3), ("foo", "new")], names=["str", "num"]),
        ).T,
    )

    with pytest.raises(ValueError):
        aggregatelevel(mdf, num=dict(new=[1, 2]), axis="no-axis")

    # append w/o conflicts
    assert_frame_equal(
        aggregatelevel(mdf, num=dict(new=[1, 2]), mode="append"),
        DataFrame(
            dict(one=[1, 1, 1, 2], two=[1, 2, 3, 3]),
            MultiIndex.from_tuples(
                [("foo", 1), ("foo", 2), ("bar", 3), ("foo", "new")],
                names=["str", "num"],
            ),
        ),
    )

    # append w/ conflicts
    assert_frame_equal(
        aggregatelevel(mdf, num={1: [1, 2]}, mode="append"),
        DataFrame(
            dict(one=[1, 1, 1], two=[3, 1, 2]),
            MultiIndex.from_tuples(
                [("bar", 3), ("foo", 1), ("foo", 2)],
                names=["str", "num"],
            ),
        ),
    )

    # return
    assert_frame_equal(
        aggregatelevel(mdf, num=dict(new=[1, 2]), mode="return"),
        DataFrame(
            dict(one=[2], two=[3]),
            MultiIndex.from_tuples([("foo", "new")], names=["str", "num"]),
        ),
    )

    with pytest.raises(ValueError):
        aggregatelevel(mdf, num=dict(new=[1, 2]), mode="bla")


def test_add_zeros_like(mdf):
    reference = MultiIndex.from_arrays(
        [["foo", "foo", "bar", "baz"], [1, 2, 3, 4], ["a", "b", "c", "d"]],
        names=["str", "num", "new"],
    )
    assert_frame_equal(
        add_zeros_like(mdf, reference),
        mdf.reindex(reference.droplevel("new"), fill_value=0),
    )

    assert_frame_equal(
        add_zeros_like(mdf, Series(0, reference)),
        mdf.reindex(reference.droplevel("new"), fill_value=0),
    )

    assert_frame_equal(add_zeros_like(mdf, reference, blub=[]), mdf)

    missing = MultiIndex.from_arrays(
        [["bar", "baz", "foo", "baz"], [2, 2, 3, 3]], names=["str", "num"]
    )
    assert_frame_equal(
        add_zeros_like(mdf, reference, num=[2, 3]),
        mdf.reindex(mdf.index.append(missing), fill_value=0),
    )

    def add_first(df):
        index = df if isinstance(df, Index) else df.index
        return assignlevel(df, first=projectlevel(index, "str").str[:1])

    mdf_w_first = add_first(mdf)
    assert_frame_equal(
        add_zeros_like(
            mdf_w_first,
            reference,
            num=[2, 3],
            derive=dict(first=add_first(Index(["foo", "bar", "baz"], name="str"))),
        ),
        mdf_w_first.reindex(mdf_w_first.index.append(add_first(missing)), fill_value=0),
    )
