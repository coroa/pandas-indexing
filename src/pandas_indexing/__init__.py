"""
Pandas indexing helper.
"""


from importlib.metadata import version as _version

from . import core, datasets
from .arithmetics import add, divide, multiply, subtract
from .core import (
    alignlevel,
    alignlevels,
    assignlevel,
    dropnalevel,
    index_names,
    projectlevel,
    semijoin,
    uniquelevel,
)
from .selectors import isin, ismatch


try:
    __version__ = _version("pandas-indexing")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
