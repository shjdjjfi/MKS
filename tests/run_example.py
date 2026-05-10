from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_pipeline import load_quarterly_macro, train_test_split_time
from src.model import CapitalAccumulationModel


def main() -> None:
    df = load_quarterly_macro()
    train, test = train_test_split_time(df)

    features = ["x_cons", "x_inv", "x_rate"]
    model = CapitalAccumulationModel(feature_names=features)

    fit = model.fit(train["K_gdp"].values, train[features].values, train["K_next"].values)
    pred = model.predict_next(test["K_gdp"].values, test[features].values)
    y_true = test["K_next"].values

    mae = mean_absolute_error(y_true, pred)
    rmse = mean_squared_error(y_true, pred) ** 0.5

    point = test[features].iloc[-1].values
    me = model.marginal_effect(point)

    out = pd.DataFrame({"date": test.index, "pred_K_next": pred, "true_K_next": y_true, "error": pred - y_true})
    out.to_csv("data/test_predictions.csv", index=False)

    print("=== Fit Summary ===")
    print(f"R2(in-sample): {fit.r2:.4f}")
    print(f"Intercept(alpha): {fit.intercept:.4f}")
    print(f"MAE(test): {mae:.4f}")
    print(f"RMSE(test): {rmse:.4f}")

    print("\n=== Comparison (first 5 test rows) ===")
    print(out.head(5).to_string(index=False))

    print("\n=== Marginal Effects at latest test point ===")
    for n, v in zip(features, me):
        print(f"dDeltaK/d{n}: {v:.6f}")

    print("\n=== Extreme Values (test period) ===")
    print(f"pred_max: {float(np.max(pred)):.4f}")
    print(f"pred_min: {float(np.min(pred)):.4f}")
    print(f"true_max: {float(np.max(y_true)):.4f}")
    print(f"true_min: {float(np.min(y_true)):.4f}")


if __name__ == "__main__":
    main()
