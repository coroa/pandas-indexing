"""Unit handling in pandas data.

Enables unit conversions based on pint's application registry (see also Notes).

By default units are expected -- as in the IAMC default format -- on a ``unit`` level on
each row, but a column-wise ``unit`` level is also supported.

Units can be handled in one of two flavours:

1. :py:func:`convert_unit` converts manually to a new unit like `convert_unit(s, "km")`

2. :py:func:`quantify` convert data to a pint pandas array which tracks units implicitly
   through arithmetics until :py:func:`dequantify` then extracts the tracked unit back
   into the multiindex level.

   While this is in theory the simpler approach, the underlying library ``pint-pandas`` [1]_
   is brittle and breaks from time to time.

Notes
-----
The pint application registry is set by :py:func:`pint.set_application_registry` or
with :py:func:`set_openscm_registry_as_default`. The latter sets the IAMC based ``openscm-units`` one [2]_.

Examples
--------
>>> import pandas_indexing as pi
>>> pi.set_openscm_registry_as_default()
>>> s = Series(
...     [7, 8],
...     MultiIndex.from_tuples([("foo", "mm"), ("bar", "m")], names=["var", "unit"]),
... )
>>> s = pi.convert_unit(s, "km")
>>> s
var  unit
bar  km      0.008000
foo  km      0.000007
dtype: float64

>>> pi.quantify(s)
var
bar    0.008
foo    7e-06
dtype: pint[kilometer]

References
----------
.. [1] https://github.com/hgrecco/pint-pandas
.. [2] https://github.com/openscm/openscm-units

See also
--------
pint.set_application_registry
"""

from typing import Callable, Mapping, Optional, Union

from pandas import DataFrame, Series

from .core import assignlevel, uniquelevel
from .types import Axis, Data
from .utils import LazyLoader, doc


INSTALL_PACKAGE_WARNING = (
    "Please install {name} via conda or pip, or use the pandas-indexing[units] variant."
)

pint_pandas = LazyLoader(
    "pint_pandas",
    globals(),
    "pint_pandas",
    INSTALL_PACKAGE_WARNING.format(name="pint_pandas"),
)
pint = LazyLoader(
    "pint", globals(), "pint", INSTALL_PACKAGE_WARNING.format(name="pint")
)


@doc(
    data="""
    data : DataFrame or Series
        DataFrame or Series to quantify\
    """,
    example_call="quantify(s)",
)
def quantify(
    data: Data,
    level: str = "unit",
    unit: Optional[str] = None,
    axis: Axis = 0,
    copy: bool = False,
) -> Data:
    """Convert columns in `data` to pint extension types to handle units.

    `pint-pandas <https://github.com/grecco/pint-pandas>`_ can only represent a single
    unit per column and is somewhat brittle.

    Parameters
    ----------\
    {data}
    unit : str, optional
        If given, assumes data is currently in this unit.
    level : str, optional
        Level of which to use the unit, by default "unit"
    axis : Axis, optional
        Axis from which to pop the `level`, by default 0
    copy : bool, optional
        Whether data should be copied, by default False

    Returns
    -------
    Data
        Data with internalized unit which stays with arithmetics

    Raises
    ------
    ValueError
        If `level` contains more than one unit

    Examples
    --------
    >>> s = Series(
    ...     [7e-3, 8],
    ...     MultiIndex.from_tuples([("foo", "m"), ("bar", "m")], names=["var", "unit"]),
    ... )
    >>> {example_call}
    var
    foo    7e-06
    bar    0.008
    dtype: pint[kilometer]

    Notes
    -----
    pint-pandas uses the pint application registry, which can be set with
    :py:func:`pint.set_application_registry` or
    :py:func:`set_openscm_registry_as_default`.

    See also
    --------
    set_openscm_registry_as_default
    dequantify
    convert_unit
    """
    if unit is None:
        unit = uniquelevel(data, level, axis)
        if len(unit) != 1:
            raise ValueError(
                f"pint-pandas can only represent homogeneous units: {', '.join(unit)}"
            )
        unit = unit[0]
        data = data.droplevel(level, axis=axis)

    return data.astype(pint_pandas.PintType(unit), copy=copy)


def format_dtype(dtype):
    if not isinstance(dtype, pint_pandas.PintType):
        return ""

    return ("{:" + dtype.ureg.default_format + "}").format(dtype.units)


def dequantify(data: Data, level: str = "unit", axis: Axis = 0, copy: bool = False):
    if isinstance(data, Series):
        unit = format_dtype(data.dtype)
        data = data.pint.magnitude
    elif isinstance(data, DataFrame):
        if axis in (0, "index"):
            unit = data.dtypes.unique()
            if len(unit) != 1:
                raise ValueError(
                    f"Only homogeneous units can be represented: {', '.join(unit)}"
                )
            unit = format_dtype(unit[0])
            data = data.apply(lambda s: s.pint.magnitude, result_type="expand", axis=1)
        elif axis in (1, "columns"):
            unit = data.dtypes.map(format_dtype)
            data = data.apply(lambda s: s.pint.magnitude, result_type="expand", axis=0)
        else:
            raise ValueError(
                f"axis can only be one of 0, 1, 'index' or 'columns', not: {axis}"
            )
    else:
        raise TypeError(
            f"data must derive from DataFrame or Series, but is {type(data)}"
        )

    return assignlevel(data, unit=unit, axis=axis)


@doc(
    data="""
    data : DataFrame or Series
        DataFrame or Series with a "unit" level\
    """,
    convert_unit_s_km='convert_unit(s, "km")',
    convert_unit_s_m_to_km='convert_unit(s, {"m": "km"})',
)
def convert_unit(
    data: Data,
    unit: Union[str, Mapping[str, str], Callable[[str], str]],
    level: Optional[str] = "unit",
    axis: Axis = 0,
):
    """Converts units in a dataframe or series.

    Parameters
    ----------\
    {data}
    unit : str or dict or function from old to new unit
        Either a single target unit or a mapping from old unit to target unit
        (a unit missing from the mapping or with a return value of None is kept)
    level : str|None, default "unit"
        Level name on ``axis``
        If None, then ``unit`` needs to be a mapping like ``{{from_unit: to_unit}}``
    axis : Axis, default 0
        Axis of unit level

    Returns
    -------
    Data
        DataFrame or Series with converted units

    Examples
    --------
    >>> s = Series(
    ...     [7, 8],
    ...     MultiIndex.from_tuples(
    ...         [("foo", "mm"), ("bar", "m")], names=["var", "unit"]
    ...     ),
    ... )
    >>> {convert_unit_s_km}
    var  unit
    bar  km      0.008000
    foo  km      0.000007
    dtype: float64

    >>> {convert_unit_s_m_to_km}
    var  unit
    bar  km      0.008
    foo  mm      7.000
    dtype: float64

    Notes
    -----
    Uses the pint application registry, which can be set with
    :py:func:`pint.set_application_registry` or
    :py:func:`set_openscm_registry_as_default`.

    See also
    --------
    set_openscm_registry_as_default
    quantify
    dequantify
    """
    if callable(unit):
        unit_map = unit
    elif isinstance(unit, dict):
        unit_map = unit.get
    else:
        unit_map = lambda u: unit

    ur = pint.get_application_registry()

    def _convert_unit(df, old_unit=None):
        if old_unit is None:
            old_unit = df.name
        new_unit = unit_map(old_unit)
        if new_unit is None:
            return df

        factor = ur.Quantity(1, old_unit).to(new_unit).m
        df = factor * df
        if level is None:
            return df
        return assignlevel(df, **{level: new_unit})

    try:
        if level is None:
            if not (isinstance(unit, dict) and len(unit) == 1):
                raise ValueError(
                    "If level is None, unit must look like {{fromunit: tounit}}"
                )
            (old_unit,) = unit.keys()
            data = _convert_unit(data, old_unit=old_unit)
        elif axis in (0, "index"):
            data = data.groupby(level=level, group_keys=False).apply(_convert_unit)
        elif axis in (1, "columns"):
            data = data.T.groupby(level=level, group_keys=False).apply(_convert_unit).T
        else:
            raise ValueError(
                f"axis can only be one of 0, 1, 'index' or 'columns', not: {axis}"
            )

        return data.__finalize__(data, "convert_unit")
    except pint.errors.DimensionalityError as exc:
        raise exc from None  # remove exception chaining


_openscm_registry = None


def get_openscm_registry(add_co2e: bool = True):
    global _openscm_registry
    if _openscm_registry is not None:
        return _openscm_registry

    import openscm_units

    if add_co2e:
        _openscm_registry = openscm_units.ScmUnitRegistry()
        _openscm_registry.add_standards()
        _openscm_registry.define("CO2e = CO2")
        _openscm_registry.define("CO2eq = CO2")
    else:
        _openscm_registry = openscm_units.unit_registry

    return _openscm_registry


def set_openscm_registry_as_default(add_co2e: bool = True):
    unit_registry = get_openscm_registry(add_co2e=add_co2e)

    app_registry = pint.get_application_registry()
    if unit_registry is not app_registry:
        pint.set_application_registry(unit_registry)

    return unit_registry


def is_unit(unit: str) -> bool:
    ur = pint.get_application_registry()
    try:
        ur.Unit(unit)
        return True
    except pint.errors.UndefinedUnitError:
        return False
