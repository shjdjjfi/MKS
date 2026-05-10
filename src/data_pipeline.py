from __future__ import annotations

import pandas as pd
from statsmodels.datasets import macrodata


def load_quarterly_macro() -> pd.DataFrame:
    """Load real US historical macro data (1959Q1-2009Q3) from statsmodels.

    K_t proxy: real GDP.
    Features x_t:
      - real consumption
      - real investment
      - real interest rate
    """
    ds = macrodata.load_pandas().data.copy()
    period_str = ds["year"].astype(int).astype(str) + "Q" + ds["quarter"].astype(int).astype(str)
    idx = pd.PeriodIndex(period_str, freq="Q")
    df = pd.DataFrame(index=idx.to_timestamp(how="end"))
    df["K_gdp"] = ds["realgdp"].values
    df["x_cons"] = ds["realcons"].values
    df["x_inv"] = ds["realinv"].values
    df["x_rate"] = ds["realint"].values
    df["K_next"] = df["K_gdp"].shift(-1)
    return df.dropna().copy()


def train_test_split_time(df: pd.DataFrame, split_date: str = "1999-12-31"):
    train = df.loc[df.index <= split_date].copy()
    test = df.loc[df.index > split_date].copy()
    return train, test
