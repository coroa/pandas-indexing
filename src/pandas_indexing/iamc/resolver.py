from __future__ import annotations

import operator
import re
from functools import reduce
from itertools import product
from typing import Any, Callable, Iterator, Sequence, TypeVar

from attrs import define, evolve, field
from pandas import DataFrame, Index, MultiIndex, Series

from .. import arithmetics
from ..core import concat
from ..selectors import isin, ismatch
from ..utils import print_list, shell_pattern_to_regex


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
        .pix.project(names)[index.names.difference(names)]  # type: ignore
        .groupby(names)
        .agg(lambda x: print_list(set(x), n=42))
    )


def maybe_parens(provenance: str) -> str:
    if any(op in provenance for op in [" + ", " - ", " * ", " / "]):
        return f"({provenance})"
    return provenance


@define
class SharedTrigger:
    active: int = field(default=0, converter=int)

    def __enter__(self):
        self.active += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.active -= 1
        return False

    def __bool__(self):
        return bool(self.active)


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
    optional_combinations: SharedTrigger = field(factory=SharedTrigger)


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

    SV = TypeVar("SV", bound="Var")

    def __repr__(self) -> str:
        return f"Var {self.provenance}\n{self.data.pix}"

    @property
    def empty(self) -> bool:
        return self.data.empty

    def index(self, levels: Sequence[str]) -> MultiIndex:
        if not set(levels).issubset(self.data.index.names):
            return MultiIndex.from_tuples([], names=levels)
        return self.data.pix.unique(levels)  # type: ignore

    def _binop(
        self: SV,
        op,
        x: SV,
        y: SV,
        provenance_maker: Callable[[str, str], str],
    ) -> SV:
        if not all(isinstance(v, Var) for v in (x, y)):
            return NotImplemented
        provenance = provenance_maker(x.provenance, y.provenance)
        return self.__class__(op(x.data, y.data, join="inner"), provenance)

    def __add__(self: SV, other: SV) -> SV:
        return self._binop(arithmetics.add, self, other, lambda x, y: f"{x} + {y}")  # type: ignore

    def __sub__(self: SV, other: SV) -> SV:
        return self._binop(
            arithmetics.sub,  # type: ignore
            self,
            other,
            lambda x, y: f"{x} - {maybe_parens(y)}",
        )

    def __mul__(self: SV, other: SV) -> SV:
        return self._binop(
            arithmetics.mul,  # type: ignore
            self,
            other,
            lambda x, y: f"{maybe_parens(x)} * {maybe_parens(y)}",
        )

    def __truediv__(self: SV, other: SV) -> SV:
        return self._binop(
            arithmetics.div,  # type: ignore
            self,
            other,
            lambda x, y: f"{maybe_parens(x)} / {maybe_parens(y)}",
        )

    def __neg__(self: SV) -> SV:
        return self.__class__(-self.data, "- " + maybe_parens(self.provenance))

    def as_df(self, **assign: str) -> DataFrame:
        return self.data.pix.assign(**(assign | dict(provenance=self.provenance)))


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

    SV = TypeVar("SV", bound="Vars")
    T = TypeVar("T", bound=Index | DataFrame | Series)

    @classmethod
    def from_data(
        cls: type[SV],
        data: DataFrame,
        value: str,
        *,
        context: Context,
        provenance: str | None = None,
    ) -> SV:
        if provenance is None:
            provenance = value
        data = data.loc[isin(**{context.level: value})].droplevel(context.level)  # type: ignore
        return cls([Var(data, provenance)] if not data.empty else [], context)

    @classmethod
    def from_additionalresolver(
        cls: type[SV],
        vars: SV,
        prefix: str,
        *,
        context: Context,
    ) -> SV:
        data = [evolve(var, provenance=f"{prefix}({var.provenance})") for var in vars]
        return cls(data, evolve(context, full_index=vars.context.full_index))

    def __repr__(self) -> str:
        index = self.index
        incomplete = not self._missing(index).empty

        if self.empty:
            return "Vars empty"

        s = (
            f"Vars for {len(index)}{'*' if incomplete else ''} scenarios:\n"
            + "\n".join(
                (
                    f"* {v.provenance} ("
                    f"{len(v.index(self.context.index))}"
                    f"{'*' if not self._missing(v).empty else ''})"
                )
                for v in self.data
            )
        )

        if len(self) == 1:
            s += f"\n\nDetails (since only a single variant):\n{self.data[0]}"
        return s

    @property
    def empty(self) -> bool:
        return not self

    @property
    def index(self) -> MultiIndex:
        if self.empty:
            return MultiIndex.from_tuples([], names=self.context.index)

        return concat(v.index(self.context.index) for v in self.data).unique()

    def __bool__(self) -> bool:
        return bool(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Var]:
        return iter(self.data)

    def __getitem__(self: SV, i: int) -> SV:
        ret = self.data[i]
        if isinstance(ret, Var):
            ret = [ret]
        return self.__class__(ret, self.context)

    class _LocIndexer:
        def __init__(self, obj):
            self._obj = obj

        def __getitem__(self, x: Any) -> Vars:
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
    def loc(self) -> _LocIndexer:
        return self._LocIndexer(self)

    @staticmethod
    def _antijoin(data: T, vars: Vars) -> T:
        return reduce(lambda d, v: d.pix.antijoin(v.data.index), vars.data, data)  # type: ignore

    def antijoin(self: SV, other: SV) -> SV:
        """Remove everything from self that is already in `other`

        Parameters
        ----------
        other : Vars
            Another set of derivations for the same variable

        Returns
        -------
        Vars
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
        assert isinstance(index, MultiIndex)
        return _summarize(index, self.context.index) if summarize else index

    def _binop(self: SV, op: Callable[[Var, Var], Var], x: SV, y: SV) -> SV:
        if not all(isinstance(v, Vars) for v in (x, y)):
            return NotImplemented

        res = self.__class__(
            [z for u, v in product(x, y) if not (z := op(u, v)).empty], self.context
        )
        if self.context.optional_combinations:
            res = res | x | y
        return res

    def __add__(self: SV, other: SV) -> SV:
        if other == 0:
            return self
        return self._binop(operator.add, self, other)

    def __radd__(self: SV, other: SV) -> SV:
        if other == 0:
            return self
        return self._binop(operator.add, other, self)

    def __sub__(self: SV, other: SV) -> SV:
        if other == 0:
            return self
        return self._binop(operator.sub, self, other)

    def __rsub__(self: SV, other: SV) -> SV:
        if other == 0:
            return self
        return self._binop(operator.sub, other, self)

    def __mul__(self: SV, other: SV) -> SV:
        if other == 1:
            return self
        return self._binop(operator.mul, self, other)

    def __rmul__(self: SV, other: SV) -> SV:
        if other == 1:
            return self
        return self._binop(operator.mul, other, self)

    def __neg__(self: SV) -> SV:
        return self.__class__([-v for v in self], self.context)

    def __or__(self: SV, other: SV | float | int) -> SV:
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

    def __ror__(self: SV, other: SV) -> SV:
        return other ^ self.antijoin(other)

    def __xor__(self: SV, other: SV) -> SV:
        if not isinstance(other, Vars):
            return NotImplemented
        return self.__class__(self.data + other.data, self.context)

    def as_df(self, **assign: str) -> DataFrame:
        return concat(var.as_df(**assign) for var in self.data)


@define
class Resolver:
    """Resolver allows to consolidate variables by composing variants.

    Usage
    -----
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

    SR = TypeVar("SR", bound="Resolver")

    @classmethod
    def from_data(
        cls: type[SR],
        data: DataFrame,
        level: str,
        values: Sequence[str] | None = None,
        index: Sequence[str] = ("model", "scenario"),
        **external_data: DataFrame,
    ) -> SR:
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

    def add(self: SR, value: str, iamc_aggregate: bool = False) -> SR:
        vars = Vars.from_data(self.data, value, context=self.context)
        if iamc_aggregate:
            vars = vars | self.iamc_aggregate(value)
        self.vars[value] = vars
        return self

    class _LocIndexer:
        def __init__(self, obj: Resolver):
            self._obj = obj

        def __getitem__(self, x: Any) -> Resolver:
            obj = self._obj
            vars = {name: z for name, vars in obj.vars.items() if (z := vars.loc[x])}
            data = obj.data.loc[x]
            context = evolve(
                obj.context, full_index=data.index.droplevel(obj.context.level).unique()
            )
            return obj.__class__(vars, data, context, obj.external_data)

    @property
    def loc(self) -> _LocIndexer:
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
                data = self.external_data[prefix]
                if isinstance(data, Resolver):
                    vars = Vars.from_additionalresolver(
                        data[rem_value], prefix=prefix, context=self.context
                    )
                else:
                    vars = Vars.from_data(
                        self.external_data[prefix],
                        rem_value,
                        provenance=value,
                        context=self.context,
                    )
            except (KeyError, ValueError):
                vars = Vars([], context=self.context)

        return vars

    def _ipython_key_completions_(self) -> list[str]:
        comps = list(self.vars)
        comps.extend(self.data.pix.unique(self.context.level).difference(comps))  # type: ignore
        for n, v in self.external_data.items():
            comps.extend(f"{n}|" + v.pix.unique(self.context.level))  # type: ignore
        return comps

    @property
    def optional_combinations(self):
        return self.context.optional_combinations

    @property
    def index(self) -> MultiIndex:
        if not self.vars:
            return MultiIndex.from_tuples([], names=self.context.index)

        return reduce(
            MultiIndex.intersection, (vars.index for vars in self.vars.values())
        )

    def __repr__(self) -> str:
        lines = []
        num_scenarios = len(self.data.pix.unique(self.context.index))  # type: ignore
        level = self.context.level

        lines.append(
            f"Resolver with data for {num_scenarios} scenarios, "
            f"and {len(self)} defined {level}s for {len(self.index)} scenarios:"
        )
        lines.extend(
            f"* {name} ({len(vars)}): "
            + ", ".join(
                str(len(var.data.pix.unique(self.context.index)))  # type: ignore
                for var in vars.data
            )
            for name, vars in self.vars.items()
        )

        existing_provenances = set(
            v.provenance for vars in self.vars.values() for v in vars
        )
        unused_values = (
            self.data.pix.unique([*self.context.index, level])  # type: ignore
            .pix.antijoin(Index(self.vars, name=level).union(existing_provenances))
            .pix.project(level)
            .value_counts()
            .loc[lambda s: s > num_scenarios // 20]
        )
        lines.append(
            f"{len(unused_values)} {level}s for more than 5% of scenarios unused:"
        )
        lines.extend(f"* {value} ({num})" for value, num in unused_values.items())
        return "\n".join(lines)

    def __setitem__(self, value: str, vars: Vars) -> Vars:
        if not isinstance(vars, Vars):
            raise TypeError(f"Expected Vars instance, found: {type(vars)}")
        self.vars[value] = vars
        return vars

    def iamc_aggregate(self, value: str, **overwrites) -> Vars:
        pattern = f"{value}|*"
        overwritten_variables = {
            name: var
            for name, var in (self.vars | overwrites).items()
            if (
                re.match(shell_pattern_to_regex(pattern), name)
                and (var.data[0].provenance != name or len(var.data) > 1)
            )
        }

        data = (
            concat(
                [
                    self.data.loc[
                        ismatch(**{self.context.level: pattern})
                    ].pix.antijoin(
                        Index(overwritten_variables, name=self.context.level)
                    ),
                    *(
                        v.as_df(**{self.context.level: n}).droplevel("provenance")
                        for n, v in overwritten_variables.items()
                    ),
                ]
            )
            .groupby(self.data.index.names.difference([self.context.level]))
            .sum()
        )
        if data.empty:
            return Vars([], self.context)

        conditions = (
            (
                " with special "
                + ", ".join(name.removeprefix(value) for name in overwritten_variables)
            )
            if overwritten_variables
            else ""
        )
        provenance = f"sum({pattern}{conditions})"

        return Vars([Var(data, provenance)], self.context)

    def add_iamc_aggregate(self: SR, value: str) -> SR:
        self[value] |= self.iamc_aggregate(value)
        return self

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
