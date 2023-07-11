.. currentmodule:: pandas_indexing

Changelog
=========

v0.2.9 (2023-07-11)
------------------------------------------------------------
* Rename pandas accessor to ``.pix`` (``.idx`` is as of now deprecated, but available
  for the time being) in :pull:`27`.
* Fix :func:`~core.projectlevel` on columns of a ``DataFrame`` :pull:`28`

v0.2.8 (2023-06-24)
------------------------------------------------------------
* Units can be converted with :func:`~units.convert_unit`, like f.ex.
  ``convert_unit(df, "km / h")`` or with ``convert_unit(df, {"m / s": "km / h"})``
  to convert only data with the ``m / s`` unit
* If the openscm-units registry is registered as pint application registry then emission
  conversion between gas species are possible under the correct contexts:

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
