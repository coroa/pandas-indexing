.. currentmodule:: pandas_indexing

Changelog
=========

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
