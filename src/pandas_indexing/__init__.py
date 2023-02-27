"""
Pandas indexing helper.
"""


from importlib.metadata import version as _version

from . import core
from .core import (
    alignlevel,
    alignlevels,
    assignlevel,
    index_names,
    isin,
    ismatch,
    projectlevel,
    semijoin,
)


try:
    __version__ = _version("pandas-indexing")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
