from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_pipeline import load_quarterly_macro
from src.model import CapitalAccumulationModel


def rolling_window_eval(df: pd.DataFrame, features: list[str], train_size: int = 80, step: int = 4) -> pd.DataFrame:
    rows = []
    n = len(df)
    for end in range(train_size, n - 1, step):
        train = df.iloc[:end].copy()
        test = df.iloc[end : min(end + step, n)].copy()

        model = CapitalAccumulationModel(features, model_type="ridge", alpha=0.8)
        model.fit(train["K_gdp"].values, train[features].values, train["K_next"].values)
        pred = model.predict_next(test["K_gdp"].values, test[features].values)
        true = test["K_next"].values

        rows.append(
            {
                "train_end": str(train.index[-1].date()),
                "test_start": str(test.index[0].date()),
                "test_end": str(test.index[-1].date()),
                "mae": float(mean_absolute_error(true, pred)),
                "rmse": float(mean_squared_error(true, pred) ** 0.5),
            }
        )
    return pd.DataFrame(rows)


def regime_label(dt: pd.Timestamp) -> str:
    crisis_start = pd.Timestamp("2007-01-01")
    crisis_end = pd.Timestamp("2009-12-31")
    return "crisis" if crisis_start <= dt <= crisis_end else "normal"


def main() -> None:
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    df = load_quarterly_macro()
    split_date = pd.Timestamp("1999-12-31")
    train = df.loc[df.index <= split_date].copy()
    test = df.loc[df.index > split_date].copy()

    features = ["x_cons", "x_inv", "x_rate"]
    model = CapitalAccumulationModel(features, model_type="ridge", alpha=0.8)
    fit = model.fit(train["K_gdp"].values, train[features].values, train["K_next"].values)

    pred = model.predict_next(test["K_gdp"].values, test[features].values)
    true = test["K_next"].values

    result = pd.DataFrame({"date": test.index, "pred": pred, "true": true})
    result["error"] = result["pred"] - result["true"]
    result["abs_error"] = result["error"].abs()
    result["regime"] = result["date"].map(regime_label)

    regime_stats = result.groupby("regime").agg(
        mae=("abs_error", "mean"),
        rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        n=("error", "size"),
    )

    rolling = rolling_window_eval(df, features, train_size=80, step=4)

    latest_point = test[features].iloc[-1].values
    me = model.marginal_effect(latest_point)

    summary_lines = [
        "# 资本积累模型自动分析报告",
        "",
        "## 总体效果",
        f"- 模型类型: {fit.model_type} (alpha=0.8, 标准化 + 滞后项 + 二次项)",
        f"- 训练集 R2: {fit.r2:.4f}",
        f"- 测试集 MAE: {mean_absolute_error(true, pred):.4f}",
        f"- 测试集 RMSE: {mean_squared_error(true, pred) ** 0.5:.4f}",
        "",
        "## 危机期 / 正常期误差",
    ]
    for idx, row in regime_stats.iterrows():
        summary_lines.append(f"- {idx}: MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, 样本数={int(row['n'])}")

    summary_lines += ["", "## 边际效应（最新测试点）"]
    for n, v in zip(features, me):
        summary_lines.append(f"- dDeltaK/d{n}: {v:.6f}")

    summary_lines += [
        "",
        "## 极值特性（测试期）",
        f"- pred_max: {float(np.max(pred)):.4f}",
        f"- pred_min: {float(np.min(pred)):.4f}",
        f"- true_max: {float(np.max(true)):.4f}",
        f"- true_min: {float(np.min(true)):.4f}",
    ]

    report_path = output_dir / "analysis_report.md"
    report_path.write_text("\n".join(summary_lines), encoding="utf-8")

    rolling.to_csv(output_dir / "rolling_metrics.csv", index=False)
    result.to_csv(output_dir / "test_predictions_improved.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(result["date"], result["true"], label="True")
    plt.plot(result["date"], result["pred"], label="Pred")
    plt.title("K_next: Pred vs True")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pred_vs_true.png", dpi=140)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(pd.to_datetime(rolling["test_end"]), rolling["rmse"])
    plt.title("Rolling RMSE")
    plt.tight_layout()
    plt.savefig(output_dir / "rolling_rmse.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
