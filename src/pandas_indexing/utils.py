"""
Utils module.

Simple utility functions not of greater interest
"""

import importlib
import re
from textwrap import dedent
from types import ModuleType
from typing import Any, Callable, TypeVar, Union

from attrs import define
from pandas import DataFrame, Index, Series
from pandas.util._decorators import doc as pandadoc

from .types import Axis


F = TypeVar("F", bound=Callable[..., Any])

# -- when a proper fix is in, delete this chunk
# this is a copy and paste of pandas 3.0.x where this function was deprecated
from collections.abc import Callable
from pandas._typing import F as pd_F
def pandadoc(*docstrings: None | str | Callable, **params: object) -> Callable[[pd_F], pd_F]:
    """
    A decorator to take docstring templates, concatenate them and perform string
    substitution on them.

    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.

    Parameters
    ----------
    *docstrings : None, str, or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.
    """

    def decorator(decorated: F) -> F:
        # collecting docstring and docstring templates
        docstring_components: list[str | Callable] = []
        if decorated.__doc__:
            docstring_components.append(dedent(decorated.__doc__))

        for docstring in docstrings:
            if docstring is None:
                continue
            if hasattr(docstring, "_docstring_components"):
                docstring_components.extend(
                    docstring._docstring_components  # pyright: ignore[reportAttributeAccessIssue]
                )
            elif isinstance(docstring, str) or docstring.__doc__:
                docstring_components.append(docstring)

        params_applied = [
            component.format(**params)
            if isinstance(component, str) and len(params) > 0
            else component
            for component in docstring_components
        ]

        decorated.__doc__ = "".join(
            [
                component
                if isinstance(component, str)
                else dedent(component.__doc__ or "")
                for component in params_applied
            ]
        )

        # error: "F" has no attribute "_docstring_components"
        decorated._docstring_components = (  # type: ignore[attr-defined]
            docstring_components
        )
        return decorated

    return decorator
# -- when a proper fix is in, delete this chunk

def doc(*docstrings: Any, **params: Any) -> Callable[[F], F]:
    """
    A decorator to take docstring templates, concatenate them and perform string substitution on them.

    This dedents docstrings before delegating to `pandas`'s `doc` decorator, 
    which performs the actual string substitution.
    """
    return pandadoc(*docstrings, **{p: dedent(v).rstrip() for p, v in params.items()})


def shell_pattern_to_regex(s):
    """
    Escape characters with specific regexp use.
    """
    return re.escape(s).replace(r"\*\*", ".*").replace(r"\*", r"[^|]*")


def print_list(x, n):
    """
    Return a printable string of a list shortened to n characters.

    Copied from pyam.utils.print_list by Daniel Huppmann, licensed under
    Apache 2.0.

    https://github.com/IAMconsortium/pyam/blob/449b77cb1c625b455e3675801477f19e99b30e64/pyam/utils.py#L599-L638
    .
    """
    # if list is empty, only write count
    if len(x) == 0:
        return "(0)"

    # write number of elements, subtract count added at end from line width
    x = [i if i != "" else "''" for i in map(str, x)]
    count = f" ({len(x)})"
    n -= len(count)

    # if not enough space to write first item, write shortest sensible line
    if len(x[0]) > n - 5:
        return "..." + count

    # if only one item in list
    if len(x) == 1:
        return f"{x[0]} (1)"

    # add first item
    lst = f"{x[0]}, "
    n -= len(lst)

    # if possible, add last item before number of elements
    if len(x[-1]) + 4 > n:
        return lst + "..." + count
    else:
        count = f"{x[-1]}{count}"
        n -= len({x[-1]}) + 3

    # iterate over remaining entries until line is full
    for i in x[1:-1]:
        if len(i) + 6 <= n:
            lst += f"{i}, "
            n -= len(i) + 2
        else:
            lst += "... "
            break

    return lst + count


def get_axis(data: Union[Index, Series, DataFrame], axis: Axis = 0):
    """
    Get axis `axis` from `data`.
    """
    if isinstance(data, Index):
        return data
    elif isinstance(data, Series):
        return data.index
    elif isinstance(data, DataFrame):
        if axis in (0, "index"):
            return data.index
        elif axis in (1, "columns"):
            return data.columns
        else:
            raise ValueError(
                f"axis can only be one of 0, 1, 'index' or 'columns', not: {axis}"
            )
    else:
        raise ValueError(
            f"data needs to be a pandas Series or DataFrame, not: {type(data)}"
        )


def quote_list(l):
    return ", ".join(f'"{s}"' for s in l)


def s(l):
    return "s" if len(l) > 1 else ""


class LazyLoader(ModuleType):
    """
    Lazily import a module, mainly to avoid pulling in large dependencies.

    `contrib`, and `ffmpeg` are examples of modules that are large and not always
    needed, and this allows them to only be loaded when they are used.

    Copied from tensorflow's agents.tf_agents.utils.lazy_loader by The TF-Agents Authors (2020), licensed under Apache 2.0
    https://github.com/tensorflow/agents/blob/737d758452990dc3c81b8aeab1a6ae4f63afa12c/tf_agents/utils/lazy_loader.py#L28-L68
    """

    def __init__(self, local_name, parent_module_globals, name, importerrormsg=None):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._importerrormsg = importerrormsg

        super().__init__(name)

    def _load(self):
        """
        Load the module and insert it into the parent's globals.
        """
        # Import the target module and insert it into the parent's namespace
        try:
            module = importlib.import_module(self.__name__)
        except ImportError:
            if self._importerrormsg is not None:
                raise ImportError(self._importerrormsg) from None
            else:
                raise

        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)


@define
class EqualIdentity:
    __array_ufunc__ = None
    __pandas_priority__ = 5000

    obj: Any

    def __eq__(self, other):
        return self.obj is other
