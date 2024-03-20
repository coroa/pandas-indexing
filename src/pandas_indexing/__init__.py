"""
Pandas indexing helper.
"""

from importlib.metadata import version as _version

from . import accessors, core, datasets
from .arithmetics import add, divide, multiply, subtract
from .core import (
    aggregatelevel,
    antijoin,
    assignlevel,
    concat,
    describelevel,
    dropnalevel,
    extractlevel,
    formatlevel,
    index_names,
    projectlevel,
    semijoin,
    to_tidy,
    uniquelevel,
)
from .selectors import All, None_, isin, ismatch
from .units import convert_unit, dequantify, quantify, set_openscm_registry_as_default


try:
    __version__ = _version("pandas-indexing")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
