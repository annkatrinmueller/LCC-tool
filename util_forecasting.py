import numpy as np
import pandas as pd

from prophet import Prophet


def prophet_forecast(
    history,
    value_col,
    years,
    *,
    start="2025-07-31",
    seed=1234,
):
    df = history[["DATE", value_col]].copy()
    df.columns = ["ds", "y"]

    model = Prophet(mcmc_samples=0)
    model.fit(df, seed=seed)

    dates = [pd.Timestamp(start)]

    for year in range(2026, 2026 + years):
        dates.extend([
            pd.Timestamp(year=year, month=1, day=31),
            pd.Timestamp(year=year, month=7, day=31),
        ])

    future = pd.DataFrame({
        "ds": dates
    })

    np.random.seed(seed)
    forecast = model.predict(future)

    return (
        forecast[
            ["ds", "yhat_lower", "yhat", "yhat_upper"]
        ]
        .rename(columns={
            "ds": "DATE",
            "yhat_lower": "Lower",
            "yhat": "Mean",
            "yhat_upper": "Upper",
        })
    )