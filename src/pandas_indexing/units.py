"""Unit handling based on pint-pandas.

Adds the two functions `quantify` and `dequantify` which convert *columns* of a
dataframe to a pint pandas extension array.

Notes
-----
Requires the optional dependency pint-pandas. If openscm-units is available their
registry is used by default.

Examples
--------
>>> quantify(df)  # where "unit" is the level name on the index

>>> dequantify(df)

References
----------
.. [1] https://github.com/hgrecco/pint-pandas
.. [2] https://github.com/openscm/openscm-units
"""

from typing import Callable, Union


try:
    import pint

    has_pint = True
except ImportError:
    has_pint = False

try:
    from pint_pandas import PintType

    has_pint_pandas = True
except ImportError:
    has_pint_pandas = False

try:
    import openscm_units

    has_openscm_units = True
except ImportError:
    has_openscm_units = False

from .core import assignlevel, uniquelevel
from .types import Axis, Data


def quantify(
    data: Data, unit=None, level: str = "unit", axis: Axis = 0, copy=False
) -> Data:
    assert (
        has_pint_pandas
    ), "pint-pandas needed for using the `quantify` and `dequantify` functions."

    if unit is None:
        unit = uniquelevel(data, level, axis)
        if len(unit) != 1:
            raise ValueError(
                f"pint-pandas can only represent homogeneous units: {', '.join(unit)}"
            )
        unit = unit[0]
        data = data.droplevel(level, axis=axis)

    return data.astype(PintType(unit), copy=copy)


def format_dtype(dtype):
    if not isinstance(dtype, PintType):
        return ""

    return ("{:" + dtype.ureg.default_format + "}").format(dtype.units)


def dequantify(data: Data, level: str = "unit", axis: Axis = 0, copy=False):
    unit = data.dtypes.unique()
    if len(unit) != 1:
        raise ValueError(
            f"Only homogeneous units can be represented: {', '.join(unit)}"
        )
    unit = format_dtype(unit[0])

    return assignlevel(data.astype(float, copy=copy), unit=unit, axis=axis)


def convert_unit(
    data: Data,
    unit: Union[str, dict[str, str], Callable[[str], str]],
    level: str = "unit",
    axis: Axis = 0,
):
    """Converts units in a dataframe or series.

    Parameters
    ----------
    data : Data
        DataFrame or Series with a "unit" level
    unit : str or dict or function from old to new unit
        Either a single target unit or a mapping from old unit to target unit
        (a unit missing from the mapping or with a return value of None is kept)
    level : str, default "unit"
        Level name on axis `axis`
    axis : Axis, default 0
        Axis of unit level

    Returns
    -------
    Data
        DataFrame or Series with converted units
    """
    if callable(unit):
        unit_map = unit
    elif isinstance(unit, dict):
        unit_map = unit.get
    else:
        unit_map = lambda u: unit

    ur = pint.get_application_registry()

    def _convert_unit(df):
        old_unit = df.name
        new_unit = unit_map(old_unit)
        if new_unit is None:
            return df

        factor = ur.Quantity(1, old_unit).to(new_unit).m
        return assignlevel(factor * df, axis=axis, **{level: new_unit})

    try:
        return (
            data.groupby(level, axis=axis, group_keys=False)
            .apply(_convert_unit)
            .__finalize__(data, "convert_unit")
        )
    except pint.errors.DimensionalityError as exc:
        raise exc from None  # remove exception chaining


_openscm_registry = None


def get_openscm_registry(add_co2e: bool = True) -> "openscm_units.ScmUnitRegistry":
    global _openscm_registry
    if _openscm_registry is not None:
        return _openscm_registry

    assert has_openscm_units, (
        "Please install openscm-units via conda or pip, or use the "
        "pandas-indexing[units] variant."
    )

    if add_co2e:
        _openscm_registry = openscm_units.ScmUnitRegistry()
        _openscm_registry.add_standards()
        _openscm_registry.define("CO2e = CO2")
        _openscm_registry.define("CO2eq = CO2")
    else:
        _openscm_registry = openscm_units.unit_registry

    return _openscm_registry


def set_openscm_registry_as_default(
    add_co2e: bool = True,
) -> "openscm_units.ScmUnitRegistry":
    unit_registry = get_openscm_registry(add_co2e=add_co2e)

    assert (
        has_pint
    ), "Install pint via conda or pip, or use the pandas-indexing[units] variant."
    app_registry = pint.get_application_registry()
    if unit_registry is not app_registry:
        pint.set_application_registry(unit_registry)

    return unit_registry


def is_unit(unit: str) -> bool:
    assert (
        has_pint
    ), "Install pint via conda or pip, or use the pandas-indexing[units] variant."
    ur = pint.get_application_registry()
    try:
        ur.Unit(unit)
        return True
    except TypeError:
        return False
