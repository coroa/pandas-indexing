.. currentmodule:: pandas_indexing

Changelog
=========

v0.3.1 (2023-09-18)
------------------------------------------------------------
* The new :func:`~core.assignlevel` argument ``ignore_index=True`` prevents the
  dataframe and series alignment which became the default in v0.3 (yesterday),
  since there are valid use cases of the old behaviour :pull:`41`

v0.3 (2023-09-17)
------------------------------------------------------------
* **BREAKING** :func:`~core.assignlevel` aligns :class:`~pandas.Series` and
  :class:`~pandas.DataFrame` before adding them to the index :pull:`38`
* Address pandas 2.1's ``FutureWarning``s and improve test coverage :pull:`40`

v0.2.10 (2023-08-31)
------------------------------------------------------------
* Add ``mode="append"`` and ``mode="return"`` arguments to
  :func:`~core.aggregatelevel`, which extend the dataframe with the aggregated
  data or return it :pull:`39`
* Add ``fail_on_reorder`` argument to :func:`~core.semijoin` to raise a
  ``ValueError`` if the resulting data is not in the order of the provided
  index (helpful in conjunction with :func:`~core.assignlevel`) :pull:`37`
* Enhance :func:`~core.concat` to also concatenate :class:`~pandas.Index` and
  :class:`~pandas.MultiIndex` objects :pull:`37`

v0.2.10-b1 (2023-07-26)
------------------------------------------------------------
* Revise :mod:`arithmetics` module:

  * Add all standard binary ops: ``add``, ``sub``, ``mul``, ``pow``, ``mod``,
    ``floordiv``, ``truediv``, ``divmod``, ``radd``, ``rsub``, ``rmul``,
    ``rpow``, ``rmod``, ``rfloordiv``, ``rtruediv``, ``rdivmod``
  * Support in-call assignment of individual levels using ``assign`` argument,
    like ``div(generation, capacity, assign=dict(variable="capacity_factor"))``
  * Add a unit-aware variant for each binary op, like
    :func:`~arithmetics.unitadd`, or :func:`~arithmetics.unitmul`, which
    updates homogeneous units automatically with the calculation

* Add ``fill_value`` argument to :func:`~core.semijoin` for filling joining gaps
* Add :func:`~core.aggregatelevel` for aggregating individual level labels; in
  :pull:`32`
* Fix :func:`~core.formatlevel` to create a simple single-level index, if only
  a single index remains :pull:`29`
* Add :func:`~core.to_tidy` for converting a time-series data-frame to tidy
  format, as expected by plotting libraries like seaborn or plotly express; in
  :pull:`31`.

v0.2.9 (2023-07-11)
------------------------------------------------------------
* Rename pandas accessor to ``.pix`` (``.idx`` is as of now deprecated, but
  available for the time being) in :pull:`27`.
* Fix :func:`~core.projectlevel` on columns of a ``DataFrame`` :pull:`28`

v0.2.8 (2023-06-24)
------------------------------------------------------------
* Units can be converted with :func:`~units.convert_unit`, like f.ex.
  ``convert_unit(df, "km / h")`` or with ``convert_unit(df, {"m / s": "km / h"})``
  to convert only data with the ``m / s`` unit
* If the openscm-units registry is registered as pint application registry then
  emission conversion between gas species are possible under the correct
  contexts:

.. code-block:: python

    from pandas_indexing import set_openscm_registry_as_default, convert_unit

    ur = set_openscm_registry_as_default()
    with ur.context("AR6GWP100"):
        df = convert_unit(df, "Mt CO2e/yr")  # or df = df.idx.convert_unit("Mt CO2e/yr")

* To use unit conversion, you should install with ``pip install "pandas-indexing[units]"``
  to pull in the optional ``pint`` and ``openscm-units`` dependencies
* For more information about unit handling, refer to
  :py:mod:`~pandas_indexing.units` or check the code added in :pull:`17`
* Documentation fixes: MyST notebook rendering from :pull:`20` and new docs for
  :func:`~core.extractlevel` in :pull:`21`.
* Bug fixes: :func:`~core.semijoin`, :func:`~core.concat` and :func:`~selectors.ismatch`
  are working again as advertised :pull:`21` and :pull:`24`.

v0.2.7 (2023-05-26)
------------------------------------------------------------
* Compatibility release to re-include Python 3.8 support and fix CI testing
* :func:`~accessors.IndexIdxAccessor.extract` gains single-level index support
* Minimal doc improvements

v0.2.6 (2023-05-25)
------------------------------------------------------------
* :func:`~core.extractlevel` can be used on non-multiindex, like
  f.ex. ``extractlevel(df, "{sector}|{gas}")`` :pull:`18`
* :func:`~selectors.isin` accepts callable filters :pull:`16`, f.ex.
  ``df.loc[isin(year=lambda s: s>2000)]``
* New function :func:`~core.concat` makes concatenation level aware :pull:`14`

v0.2.5 (2023-05-04)
------------------------------------------------------------
* :func:`~core.formatlevel` and :func:`~core.extractlevel` (or their equivalents
  :meth:`~accessors.DataFrameIdxAccessor.format` and
  :meth:`~accessors.DataFrameIdxAccessor.extract`) make it easy to combine or split
  index levels using format-string like templates; check examples in the guide
  (:ref:`Selecting data`) :pull:`13`
* :py:func:`~core.describelevel` superseeds the as-of-now deprecated
  :py:func:`~core.summarylevel` :pull:`11`

v0.2.4 (2023-05-03)
------------------------------------------------------------

* Paper-bag release: Fix new accessors :py:func:`~accessors.IndexIdxAccessor.unique` and
  :py:func:`~accessors.IndexIdxAccessor.__repr__` and improve tests to catch trivial
  errors like these earlier :pull:`10`

v0.2.3 (2023-05-03)
------------------------------------------------------------

* :py:func:`~core.uniquelevel` or ``.idx.unique`` returns the unique values of one
  or multiple levels. :pull:`8`
* :py:func:`~core.summarylevel` creates a string summarizing the index levels and their
  values. Can also be accessed as ``df.idx`` or ``index.idx`` :pull:`9`

v0.2.2 (2023-05-02)
------------------------------------------------------------

* :py:func:`~core.assignlevel` takes labels from an optional positional dataframe :pull:`5`
* Add :py:func:`~core.dropnalevel` to remove missing index entries :pull:`4`, :pull:`6`

v0.2.1 (2023-04-08)
------------------------------------------------------------

* Restore compatibility with python 3.8
* Improve typing and add tests for :py:func:`~selectors.isin` and
  :py:func:`~selectors.ismatch`

v0.2 (2023-04-07)
------------------------------------------------------------

* :py:func:`~selectors.isin` and :py:func:`~selectors.ismatch` are now callable objects,
  which can be composed with the standard ``~``, ``&`` and ``|`` operators to more
  complex queries
* :py:func:`~arithmetics.add`, :py:func:`~arithmetics.subtract`,
  :py:func:`~arithmetics.multiply` and :py:func:`~arithmetics.divide` in the new
  :py:mod:`arithmetics` module extend the standard pandas operations with ``join`` and
  other arguments known from :py:meth:`pandas.DataFrame.align`.
  They are also available from the :py:mod:`idx accessor <accessors>`.
* Both additions were introduced in :pull:`3`

v0.1.2 (2023-02-27)
------------------------------------------------------------

* Add usage guide to documentation
* Fix :py:func:`~core.semijoin` method

v0.1.1 (2023-02-27)
------------------------------------------------------------

* Clean up documentation and switch theme (:pull:`1`)
* Add :py:func:`~core.semijoin` method (:pull:`2`)
* Introduce pandas accessor :py:mod:`idx <accessors>` on :py:class:`~pandas.DataFrame`,
  :py:class:`~pandas.Series` and :py:class:`~pandas.Index`

v0.1 (2023-02-23)
------------------------------------------------------------

* Initial release
