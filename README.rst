pandas-indexing helper
======================

.. image:: https://github.com/coroa/pandas-indexing/workflows/ci/badge.svg?branch=main
    :target: https://github.com/coroa/pandas-indexing/actions?workflow=ci
    :alt: CI

.. image:: https://codecov.io/gh/coroa/pandas-indexing/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/coroa/pandas-indexing
    :alt: Codecov

.. image:: https://img.shields.io/readthedocs/pandas-indexing/latest?label=Read%20the%20Docs
    :target: https://pandas-indexing.readthedocs.io/en/latest/
    :alt: Read the Docs

.. image:: https://img.shields.io/pypi/v/pandas-indexing
    :target: https://pypi.org/project/pandas-indexing/
    :alt: PyPI


Summary
-------

``pandas-indexing`` is a helpers package to make interacting with pandas multi-indices less
painful. It contains functions, that

* filter easily into multi indices: ``isin``, ``ismatch``
* add or update levels in a multiindex: ``assignlevel``
* select one or multiple specific levels: ``projectlevel``

Usage
-----

Given you have a time-series like dataframe with several multi index levels, like ``model``, ``scenario``, ``variable``:
then you can select a subset with:

.. code:: python

    df.loc[isin(model="m1", scenario=["s1", "s2"])]

or with shell like glob-patterns:

.. code:: python

    df.loc[
        ismatch(
            model="REMIND*", variable="Emissions|**", unit=["Mt CO2/yr", "kt N2O/yr"]
        )
    ]

You can overwrite index levels:

.. code:: python

    assignlevel(df, selected=1)

or project your data to only a few desired index levels:

.. code:: python

    projectlevel(df, ["model", "scenario"])


All commands are described in detail in the API reference in `documentation`_ and are
togehter with introductions for installing and using this package, but they are mostly
bare-bones atm.

Issues and Discussions
----------------------

As usual for any GitHub-based project, raise an `issue`_ if you find any bug or
want to suggest an improvement, or open a discussion if you want to discuss.


.. _PyPI: https://pypi.org
.. _latest branch: https://github.com/coroa/pandas-indexing/tree/latest
.. _master branch: https://github.com/coroa/pandas-indexing/tree/master
.. _tox: https://tox.readthedocs.io/en/latest/
.. _ReadTheDocs: https://readthedocs.org/
.. _issue: htts://github.com/coroa/pandas-indexing/issues/new
.. _documentation: https://pandas-indexing.readthedocs.io/en/latest/
