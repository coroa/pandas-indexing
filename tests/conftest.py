import pytest
from pandas import DataFrame, Index, MultiIndex, Series


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
