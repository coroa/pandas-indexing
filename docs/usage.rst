Guide
=====

To use ``pandas_indexing``::

    import pandas_indexing

Given you have a time-series like dataframe with several multi index levels, like ``model``, ``scenario``, ``variable``:


    .. df = pd.DataFrame(
    ..     [
    ..         [50, 25, 20],
    ..         [50, 20, 0]
    ..     ],
    ..     index=pd.MultiIndex.from_tuples(
    ..         [
    ..             ("m1", "s1", "emi"),
    ..             ("m1", "s2", "emi")
    ..         ],
    ..         names=["model", "scenario", "variable"]
    ..     ),
    ..     columns=pd.Index([2020, 2040, 2060], name="year")
    .. )

.. raw:: html

    <table border="1" class="dataframe">
    <thead>
        <tr style="text-align: right;">
        <th></th>
        <th></th>
        <th>year</th>
        <th>2020</th>
        <th>2040</th>
        <th>2060</th>
        </tr>
        <tr>
        <th>model</th>
        <th>scenario</th>
        <th>variable</th>
        <th></th>
        <th></th>
        <th></th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <th rowspan="2" valign="top">m1</th>
        <th>s1</th>
        <th>emi</th>
        <td>50</td>
        <td>25</td>
        <td>20</td>
        </tr>
        <tr>
        <th>s2</th>
        <th>emi</th>
        <td>50</td>
        <td>20</td>
        <td>0</td>
        </tr>
    </tbody>
    </table>

then you can select a subset with:

.. code:: python

    df.loc[isin(scenario="s1")]

.. raw:: html

    <table border="1" class="dataframe">
    <thead>
        <tr style="text-align: right;">
        <th></th>
        <th></th>
        <th>year</th>
        <th>2020</th>
        <th>2040</th>
        <th>2060</th>
        </tr>
        <tr>
        <th>model</th>
        <th>scenario</th>
        <th>variable</th>
        <th></th>
        <th></th>
        <th></th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <th>m1</th>
        <th>s1</th>
        <th>emi</th>
        <td>50</td>
        <td>25</td>
        <td>20</td>
        </tr>
    </tbody>
    </table>

or even on multiple levels or for multiple values:

.. code:: python

    df.loc[isin(scenario=["s1", "s2"], variable="emi")]
