from typing import Literal, TypeVar, Union

from pandas import DataFrame, Index, Series


Axis = Literal[0, 1, "index", "columns"]
Data = Union[Series, DataFrame]
T = TypeVar("T", bound=Union[Index, DataFrame, Series])
S = TypeVar("S", bound=Union[DataFrame, Series])
