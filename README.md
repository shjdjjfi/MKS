# EC Theory -> Python Capital Accumulation Project

这个项目把 `EC.tex` 中的资本积累理论转换为可执行 Python 项目。

## 理论映射

实现的核心公式：

\[
K_{t+1}=K_t+\mathcal{A}(K_t,x_t)
\]

其中使用二阶近似：

\[
\Delta K_t=\alpha+\beta^\top x_t+\frac{1}{2}x_t^\top Hx_t+\varepsilon_t
\]

通过“线性项 + 二次交互项”构建设计矩阵，并用 OLS 拟合。

## 数据（真实历史）

测试样例使用 `statsmodels` 自带美国宏观历史数据库 `macrodata`（1959Q1-2009Q3）。
- `K_t`: real GDP (`realgdp`)
- `x_cons`: real consumption (`realcons`)
- `x_inv`: real investment (`realinv`)
- `x_rate`: real interest rate (`realint`)

这是真实历史统计序列，不是随机模拟数据。

## 运行

```bash
pip install -r requirements.txt
PYTHONPATH=. python tests/run_example.py
```

## 输出内容

脚本会输出：
1. 训练拟合效果（R²）
2. 测试误差（MAE / RMSE）
3. 预测值与真实历史值前5行对比
4. 边际效应（\(\partial \Delta K/\partial x_j\)）
5. 测试期极值（预测 max/min 与真实 max/min）

并生成 `data/test_predictions.csv`。

## 建模特性（与你的 EC.tex 对应）

- **多特征资本积累**：\(x_t\) 可扩展到更多变量。
- **非线性耦合**：通过二次项刻画互补/替代关系。
- **边际效应**：数值梯度近似，便于解释单调性。
- **极值分析**：检验模型对历史高低点的响应能力。

