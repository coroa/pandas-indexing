from __future__ import annotations

import operator
from contextlib import contextmanager
from functools import reduce
from itertools import product
from typing import Any, Callable, Sequence, TypeVar

from attrs import define, evolve
from pandas import DataFrame, Index, MultiIndex, Series

from .. import arithmetics
from ..core import concat, isin
from ..utils import print_list


def _summarize(index: MultiIndex, names: Sequence[str]) -> DataFrame:
    """Summarize unique level values grouped by `names` levels.

    Parameters
    ----------
    index : MultiIndex
        Index to summarize
    names : Sequence[str]
        Levels to group by values by

    Returns
    -------
    DataFrame
        Summary frame
    """
    return (
        index.to_frame()
        .pix.project(names)[index.names.difference(names)]
        .groupby(names)
        .agg(lambda x: print_list(set(x), n=42))
    )


def maybe_parens(provenance: str) -> str:
    if any(op in provenance for op in [" + ", " - ", " * ", " / "]):
        return f"({provenance})"
    return provenance


@define
class Context:
    """Context shared between all Vars instances in a Resolver.

    Notes
    -----
    Not to be instantiated by the user herself
    """

    level: str
    full_index: MultiIndex
    columns: Index
    index: list[str]
    optional_combinations: bool = False


SelfVar = TypeVar("SelfVar", bound="Var")


@define
class Var:
    """Instance for a single variant.

    Attributes
    ----------
    data : DataFrame
        Calculated data
    provenance : str
        Formula for how the variant was calculated

    Notes
    -----
    User does not interact with individual `Var` instances, instead she only ever holds
    `Vars` instances.
    """

    data: DataFrame
    provenance: str

    def __repr__(self) -> str:
        return f"Var {self.provenance}\n{self.data.pix}"

    @property
    def empty(self) -> bool:
        return self.data.empty

    def index(self, levels: Sequence[str]) -> MultiIndex:
        if not set(levels).issubset(self.data.index.names):
            return MultiIndex.from_tuples([], names=levels)
        return self.data.pix.unique(levels)

    def _binop(
        self, op, x: SelfVar, y: SelfVar, provenance_maker: Callable[[str, str], str]
    ) -> SelfVar:
        if not all(isinstance(v, Var) for v in (x, y)):
            return NotImplemented
        provenance = provenance_maker(x.provenance, y.provenance)
        return self.__class__(op(x.data, y.data, join="inner"), provenance)

    def __add__(self, other: SelfVar) -> SelfVar:
        return self._binop(arithmetics.add, self, other, lambda x, y: f"{x} + {y}")

    def __sub__(self, other: SelfVar) -> SelfVar:
        return self._binop(
            arithmetics.sub, self, other, lambda x, y: f"{x} - {maybe_parens(y)}"
        )

    def __mul__(self, other: SelfVar) -> SelfVar:
        return self._binop(
            arithmetics.mul,
            self,
            other,
            lambda x, y: f"{maybe_parens(x)} * {maybe_parens(y)}",
        )

    def __truediv__(self, other: SelfVar) -> SelfVar:
        return self._binop(
            arithmetics.div,
            self,
            other,
            lambda x, y: f"{maybe_parens(x)} / {maybe_parens(y)}",
        )

    def as_df(self, **assign: str) -> DataFrame:
        return self.data.pix.assign(**(assign | dict(provenance=self.provenance)))


SelfVars = TypeVar("SelfVars", bound="Vars")
T = TypeVar("T", bound=Index | DataFrame | Series)


@define
class Vars:
    """`Vars` holds several derivations of a variable from data in a `Resolver`

    Attributes
    ----------
    data : list of Var
        Disjunct derivations of a single variable
    context : Context
        Shared context from the resolver
    index : MultiIndex for which any derivation has data

    Notes
    -----
    `Vars` are created with a Resolver and are to be composed with one another.

    Example
    -------
    >>> r = Resolver.from_data(co2emissions, level="sector")
    >>> energy = r["Energy"] | (r["Energy|Supply"] + r["Energy|Demand"])
    >>> r["Energy and Industrial Processes"] | (energy + r["Industrial Processes"])
    """

    data: list[Var]
    context: Context

    @classmethod
    def from_data(
        cls,
        data: DataFrame,
        value: Any,
        *,
        context: Context,
        provenance: str | None = None,
    ) -> SelfVars:
        if provenance is None:
            provenance = value
        data = data.loc[isin(**{context.level: value})].droplevel(context.level)
        return cls([Var(data, provenance)] if not data.empty else [], context)

    def __repr__(self) -> str:
        index = self.index
        incomplete = not self._missing(index).empty
        return (
            f"Vars for {len(index)}{'*' if incomplete else ''} scenarios:\n"
            + "\n".join(
                (
                    f"* {v.provenance} ("
                    f"{len(v.index(self.context.index))}"
                    f"{'*' if not self._missing(v).empty else ''})"
                )
                for v in self.data
            )
            + "\n"
        )

    @property
    def index(self) -> MultiIndex:
        if not self.data:
            return MultiIndex.from_tuples([], names=self.context.index)

        return concat(v.index(self.context.index) for v in self.data).unique()

    def __bool__(self) -> bool:
        return bool(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i: int) -> SelfVars:
        ret = self.data[i]
        if isinstance(ret, Var):
            ret = [ret]
        return self.__class__(ret, self.context)

    class _LocIndexer:
        def __init__(self, obj: SelfVars):
            self._obj = obj

        def __getitem__(self, x: Any) -> SelfVars:
            obj = self._obj
            return obj.__class__(
                [
                    var.__class__(z, var.provenance)
                    for var in obj.data
                    if not (z := var.data.loc[x]).empty
                ],
                obj.context,
            )

    @property
    def loc(self) -> SelfVars:
        return self._LocIndexer(self)

    @staticmethod
    def _antijoin(data: T, vars: SelfVars) -> T:
        return reduce(lambda d, v: d.pix.antijoin(v.data.index), vars.data, data)

    def antijoin(self, other: SelfVars) -> SelfVars:
        """Remove everything from self that is already in `other`

        Parameters
        ----------
        other : SelfVars
            Another set of derivations for the same variable

        Returns
        -------
        SelfVars
            Subset of self that is not already provided by `other`
        """
        return self.__class__(
            [
                z
                for var in self.data
                if not (z := Var(self._antijoin(var.data, other), var.provenance)).empty
            ],
            self.context,
        )

    def _missing(self, partial: bool | Var | MultiIndex = False) -> MultiIndex:
        full_index = self.context.full_index
        if isinstance(partial, Var):
            return full_index.join(
                partial.index(self.context.index), how="inner"
            ).pix.antijoin(partial.data.index)

        if isinstance(partial, MultiIndex):
            full_index = full_index.join(partial, how="inner")
        elif partial:
            full_index = full_index.join(self.index, how="inner")
        return self._antijoin(full_index, self)

    def missing(
        self, partial: bool = True, summarize: bool = True
    ) -> DataFrame | MultiIndex:
        index = self._missing(partial)
        return _summarize(index, self.context.index) if summarize else index

    def existing(self, summarize: bool = True) -> DataFrame | MultiIndex:
        index = concat(var.data.index for var in self.data)
        return _summarize(index, self.context.index) if summarize else index

    def _binop(self, op, x: SelfVars, y: SelfVars) -> SelfVars:
        if not all(isinstance(v, Vars) for v in (x, y)):
            return NotImplemented

        res = self.__class__(
            [z for u, v in product(x, y) if not (z := op(u, v)).empty], self.context
        )
        if self.context.optional_combinations:
            res = res | x | y
        return res

    def __add__(self, other: SelfVars) -> SelfVars:
        if other == 0:
            return self
        return self._binop(operator.add, self, other)

    def __radd__(self, other: SelfVars) -> SelfVars:
        if other == 0:
            return self
        return self._binop(operator.add, other, self)

    def __sub__(self, other: SelfVars) -> SelfVars:
        if other == 0:
            return self
        return self._binop(operator.sub, self, other)

    def __rsub__(self, other: SelfVars) -> SelfVars:
        if other == 0:
            return self
        return self._binop(operator.sub, other, self)

    def __mul__(self, other: SelfVars) -> SelfVars:
        if other == 1:
            return self
        return self._binop(operator.mul, self, other)

    def __rmul__(self, other: SelfVars) -> SelfVars:
        if other == 1:
            return self
        return self._binop(operator.mul, other, self)

    def __or__(self, other: SelfVars | float | int) -> SelfVars:
        if isinstance(other, (float, int)):
            provenance = str(other)
            other_index = self._missing()
            if other_index.empty:
                return self
            other = DataFrame(
                other,
                index=other_index,
                columns=self.context.columns,
            )
            return self.__class__(self.data + [Var(other, provenance)], self.context)

        return self ^ other.antijoin(self)

    def __ror__(self, other: SelfVars) -> SelfVars:
        return other ^ self.antijoin(other)

    def __xor__(self, other: SelfVars) -> SelfVars:
        if not isinstance(other, Vars):
            return NotImplemented
        return self.__class__(self.data + other.data, self.context)

    def as_df(self, **assign: str) -> DataFrame:
        return concat(var.as_df(**assign) for var in self.data)


SelfResolver = TypeVar("SelfResolver", bound="Resolver")


@define
class Resolver:
    """Resolver allows to consolidate variables by composing variants.

    Examples
    --------
    >>> co2emissions = ar6.loc[isin(gas="CO2")]
    >>> r = Resolver.from_data(co2emissions, "sector", ["AFOLU", "Energy"])
    >>> r["Energy"] |= r["Energy|Demand"] + r["Energy|Supply"]
    >>> r["AFOLU"] |= r["AFOLU|Land"] + r["AFOLU|Agriculture"]
    >>> r.as_df()
    """

    vars: dict[str, Vars]
    data: DataFrame
    context: Context  # context is shared with all Vars created
    external_data: dict[str, DataFrame]

    @classmethod
    def from_data(
        cls,
        data: DataFrame,
        level: str,
        values: Sequence[str] | None = None,
        index: Sequence[str] = ("model", "scenario"),
        **external_data: DataFrame,
    ) -> SelfResolver:
        context = Context(
            level,
            full_index=data.index.droplevel(level).unique(),
            columns=data.columns,
            index=list(index),
        )
        inst = cls({}, data, context, external_data)
        if values is not None:
            for value in values:
                inst.add(value)
        return inst

    def add(self, value: str) -> Vars:
        self.vars[value] = vars = Vars.from_data(self.data, value, context=self.context)
        return vars

    class _LocIndexer:
        def __init__(self, obj: SelfResolver):
            self._obj = obj

        def __getitem__(self, x: Any) -> SelfResolver:
            obj = self._obj
            vars = {name: z for name, vars in obj.vars.items() if (z := vars.loc[x])}
            data = obj.data.loc[x]
            context = evolve(
                obj.context, full_index=data.index.droplevel(obj.context.level).unique()
            )
            return obj.__class__(vars, data, context, obj.external_data)

    @property
    def loc(self) -> SelfResolver:
        return self._LocIndexer(self)

    def __len__(self) -> int:
        return len(self.vars)

    def __getitem__(self, value: str) -> Vars:
        vars = self.vars.get(value)
        if vars is not None:
            return vars

        vars = Vars.from_data(self.data, value, context=self.context)
        if not vars:
            try:
                # get variable from additional data
                prefix, rem_value = value.split("|", 1)
                vars = Vars.from_data(
                    self.external_data[prefix],
                    rem_value,
                    provenance=value,
                    context=self.context,
                )
            except (KeyError, ValueError):
                raise KeyError(
                    f"{value} is not a {self.context.level} in data or external_data"
                ) from None

        return vars

    def _ipython_key_completions_(self) -> list[str]:
        comps = list(self.vars)
        comps.extend(self.data.pix.unique(self.context.level).difference(comps))
        for n, v in self.external_data.items():
            comps.extend(f"{n}|" + v.pix.unique(self.context.level))
        return comps

    @property
    def index(self) -> MultiIndex:
        if not self.vars:
            return MultiIndex.from_tuples([], names=self.context.index)

        return reduce(
            MultiIndex.intersection, (vars.index for vars in self.vars.values())
        )

    def __repr__(self) -> str:
        num_scenarios = len(self.data.pix.unique(self.context.index))
        level = self.context.level

        s = (
            f"Resolver with data for {num_scenarios} scenarios, "
            f"and {len(self)} defined {level}s for {len(self.index)} scenarios:\n"
        )
        for name, vars in self.vars.items():
            s += (
                f"* {name} ({len(vars)}): "
                + ", ".join(
                    str(len(var.data.pix.unique(self.context.index)))
                    for var in vars.data
                )
                + "\n"
            )

        existing_provenances = [
            v.provenance for vars in self.vars.values() for v in vars
        ]

        unused_values = (
            self.data.pix.unique([*self.context.index, level])
            .pix.antijoin(Index(self.vars, name=level).union(existing_provenances))
            .pix.project(level)
            .value_counts()
            .loc[lambda s: s > num_scenarios // 20]
        )
        s += f"{len(unused_values)} {level}s for more than 5% of scenarios unused:\n"
        for value, num in unused_values.items():
            s += f"* {value} ({num})\n"
        return s

    def __setitem__(self, value: str, vars: Vars) -> Vars:
        if not isinstance(vars, Vars):
            raise TypeError(f"Expected Vars instance, found: {type(vars)}")
        self.vars[value] = vars
        return vars

    @contextmanager
    def optional_combinations(self):
        active = self.context.optional_combinations
        try:
            self.context.optional_combinations = True
            yield
        finally:
            self.context.optional_combinations = active

    def as_df(self, only_consistent: bool = True) -> DataFrame:
        if only_consistent:
            index = self.index

            def maybe_consistent(df):
                return df.pix.semijoin(index, how="right")
        else:

            def maybe_consistent(df):
                return df

        return concat(
            vars.as_df(**{self.context.level: name}).pipe(maybe_consistent)
            for name, vars in self.vars.items()
        )
