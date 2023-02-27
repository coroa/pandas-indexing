"""
Test datasets for simplifying the tutorial.
"""

from importlib.resources import open_text

import pandas as pd


def remindhighre_power():
    """
    Reads IAMC power sector data from REMIND's HighRE IMP scenario for AR6.

    Returns
    -------
    pd.DataFrame
        Time-series like power sector data

    .. [1] Luderer, G., Madeddu, S., Merfort, L. et al. Impact of declining renewable
       energy costs on electrification in low-emission scenarios. Nat Energy 6, 32–42
       (2022). https://doi.org/10.1038/s41560-021-00937-z
    """
    return pd.read_csv(
        open_text(__name__, "remindhighre_power.csv"), index_col=list(range(5))
    ).rename(columns=int)
