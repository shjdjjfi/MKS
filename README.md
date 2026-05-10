# EC Theory -> Python Capital Accumulation Project

这个项目把 `EC.tex` 中的资本积累理论转换为可执行 Python 项目，并提供改进版自动评估脚本。

## 改进点

- 加入滞后项（特征的 1 阶滞后）
- 标准化（StandardScaler）
- 可选 OLS / Ridge / Lasso
- Rolling-window 时间序列验证
- 危机期/正常期分段误差
- 自动生成中文分析报告与图表到本地 `artifacts/`

> 默认不会将图表和产物推送到 GitHub（通过 `.gitignore` 忽略）。

## 运行

```bash
pip install -r requirements.txt
PYTHONPATH=. python tests/run_improved_example.py
```

## 输出（本地）

- `artifacts/analysis_report.md`
- `artifacts/rolling_metrics.csv`
- `artifacts/test_predictions_improved.csv`
- `artifacts/pred_vs_true.png`
- `artifacts/rolling_rmse.png`

