import pytest
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal

from pandas_indexing import assignlevel, convert_unit, set_openscm_registry_as_default
from pandas_indexing.units import has_openscm_units, has_pint, has_pint_pandas, is_unit


needs_pint = pytest.mark.skipif(not has_pint, reason="Needs pint package")
needs_openscm_units = pytest.mark.skipif(
    not has_openscm_units, reason="Needs openscm-units package"
)
needs_pint_pandas = pytest.mark.skipif(
    not has_pint_pandas, reason="Needs pint-pandas package"
)


@pytest.fixture
def mseru(mseries):
    return assignlevel(mseries, unit=["m", "mm", "m"])


@pytest.fixture
def mdfu(mdf):
    return assignlevel(mdf, unit=["m", "mm", "m"])


@needs_pint
def test_convert_unit(mdfu: DataFrame, mseru: Series):
    assert_frame_equal(
        convert_unit(mdfu, "km"),
        assignlevel(mdfu.multiply([1e-3, 1e-6, 1e-3], axis=0), unit="km"),
        check_like=True,  # convert_unit reorders the index
    )

    assert_frame_equal(
        convert_unit(mdfu, {"m": "km", "mm": "m"}),
        assignlevel(mdfu * 1e-3, unit=["km", "m", "km"]),
        check_like=True,
    )

    assert_series_equal(convert_unit(mseru, {"m": "km"}, level=None), mseru * 1e-3)

    with pytest.raises(ValueError):
        convert_unit(mseru, "km", level=None)

    assert_series_equal(
        convert_unit(mseru, lambda u: None if u == "m" else "km"),
        assignlevel(mseru.multiply([1, 1e-6, 1], axis=0), unit=["m", "km", "m"]),
        check_like=True,
    )


@needs_openscm_units
def test_set_openscm_registry_as_default():
    import pint

    ur = set_openscm_registry_as_default()
    assert isinstance(ur, pint.UnitRegistry)
    assert hasattr(ur, "CO2e")

    ur2 = set_openscm_registry_as_default()
    assert ur is ur2


@needs_pint
def test_is_unit():
    assert is_unit("m")
    assert not is_unit("blub")
