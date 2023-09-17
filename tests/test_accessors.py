"""
Performs general tests.
"""

from textwrap import dedent

import pytest
from numpy import nan
from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

import pandas_indexing.accessors  # noqa: F401


def test_assign_index(midx: MultiIndex):
    """
    Checks scalar setting of new and old levels.
    """

    # Check for adding to multilevel
    assert_index_equal(
        midx.pix.assign(new=[1, 2, 3]),
        MultiIndex.from_arrays(
            [midx.get_level_values(0), midx.get_level_values(1), [1, 2, 3]],
            names=["str", "num", "new"],
        ),
    )

    # Check adding from dataframe (auto-alignment will switch first and second line)
    assert_index_equal(
        midx.pix.assign(
            DataFrame({"new": [1, 2, 3], "new2": 2}, index=Index([2, 1, 3], name="num"))
        ),
        MultiIndex.from_arrays(
            [midx.get_level_values(0), midx.get_level_values(1), [2, 1, 3], [2, 2, 2]],
            names=["str", "num", "new", "new2"],
        ),
    )


def test_assign_dataframe(mdf: DataFrame):
    """
    Checks setting dataframes on both axes.
    """
    assert_frame_equal(
        mdf.pix.assign(new=2),
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
        mdf.pix.assign(new=2, axis=1),
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

    assert_index_equal(midx.pix.dropna(subset=["num", "num2"], how="all"), midx[[1, 2]])


def test_project(midx, mdf):
    assert_frame_equal(
        mdf.pix.project("str"), mdf.set_axis(mdf.index.get_level_values("str"))
    )
    assert_index_equal(
        midx.pix.project(["num", "str"]),
        MultiIndex.from_arrays(
            [mdf.index.get_level_values("num"), mdf.index.get_level_values("str")]
        ),
    )


def test_repr(mdf):
    assert (
        str(mdf.pix)
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


def test_unique(mdf):
    assert_index_equal(mdf.pix.unique("str"), Index(["foo", "bar"], name="str"))


def test_convert_unit(mseries):
    assert_series_equal(
        mseries.pix.convert_unit({"m": "km"}, level=None),
        mseries * 1e-3,
    )


def test_idx_deprecation(mdf, mseries, midx):
    for obj in [mdf, mseries, midx]:
        with pytest.deprecated_call():
            obj.idx


def test_to_tidy(mseries):
    assert_frame_equal(
        mseries.pix.to_tidy(),
        DataFrame(dict(str=["foo", "foo", "bar"], num=[1, 2, 3], value=[1, 2, 3])),
    )


def test_aggregate(mdf):
    assert_frame_equal(
        mdf.pix.aggregate(num=dict(new=[1, 2])),
        DataFrame(
            dict(one=[1, 2], two=[3, 3]),
            MultiIndex.from_tuples([("bar", 3), ("foo", "new")], names=["str", "num"]),
        ),
    )
