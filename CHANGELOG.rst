.. currentmodule:: pandas_indexing

Changelog
=========

v0.2.6 (2023-05-25)
------------------------------------------------------------
* :func:`~core.extractlevel` can be used on non-multiindex, like
  f.ex. ``extractlevel(df, "{sector}|{gas}")`` :pull:`17`
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
