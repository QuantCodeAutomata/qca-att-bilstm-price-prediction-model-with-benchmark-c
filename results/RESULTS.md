# Experiment Results

_Generated: 2026-06-26 14:34 UTC_

---
## Experiment 1 — Att-BiLSTM Price Prediction

### Gold

| Model | RMSE | MAE | MAPE (%) | R² |
|---|---|---|---|---|
| Att-BiLSTM | 34.94 | 29.26 | 1.6435 | 0.9399 |
| LSTM | 25.23 | 17.74 | 1.0142 | 0.9687 |
| GRU | 21.45 | 14.63 | 0.8345 | 0.9774 |
| BiLSTM | 22.00 | 16.06 | 0.9126 | 0.9762 |
| XGBoost | 301.30 | 268.03 | 14.7201 | -3.4663 |
| ARIMA | 305.29 | 272.14 | 14.9525 | -3.5855 |
| HMM | 191.83 | 160.09 | 8.7659 | -0.8104 |
| Holt-Winters | 245.19 | 215.27 | 11.8121 | -1.9577 |

**Best model by RMSE:** GRU

### Bitcoin

| Model | RMSE | MAE | MAPE (%) | R² |
|---|---|---|---|---|
| Att-BiLSTM | 5238.95 | 3100.19 | 8.8770 | 0.9112 |
| LSTM | 1534.69 | 889.09 | 3.2610 | 0.9924 |
| GRU | 2636.53 | 1829.81 | 7.3995 | 0.9775 |
| BiLSTM | 2024.80 | 1396.23 | 6.7583 | 0.9867 |
| XGBoost | 19822.65 | 12000.11 | 29.2431 | -0.2707 |
| ARIMA | 22916.27 | 14983.18 | 43.4855 | -0.6983 |
| HMM | 20637.69 | 13043.06 | 36.1661 | -0.3774 |
| Holt-Winters | 21383.03 | 13624.38 | 38.0093 | -0.4787 |

**Best model by RMSE:** LSTM

---
## Experiment 2 — Apriori Rise/Fall Pattern Analysis

| Asset | u_n (rises) | u_n (falls) | M_u0.9 | M_u0.5 | M_d0.1 |
|---|---|---|---|---|---|
| Gold | 4 | 3 | 0.01409 | 0.00480 | -0.01565 |
| Bitcoin | 4 | 3 | 0.05876 | 0.01570 | -0.06666 |

M_i = (p_i − p_{i−1}) / p_i (current price as denominator)

---
## Experiment 3 — Exponential Position Sizing

| Asset | u_n | a | b | R² | Allocations y_i ($) |
|---|---|---|---|---|---|
| Gold | 3 | 0.9205 | 0.0828 | 1.0000 | $153.07, $166.29, $180.64 |
| Bitcoin | 3 | 0.7066 | 0.3473 | 1.0000 | $113.17, $160.16, $226.67 |

Sum of allocations verified to equal PRO=$500 per asset.

---
## Experiment 4 — Trading Strategy Simulation

### Base Simulation Results

| Asset | Initial ($) | Final ($) | Return (%) | Trades |
|---|---|---|---|---|
| Gold | 500 | 417.93 | -16.41 | 193 |
| Bitcoin | 500 | 821.89 | 64.38 | 197 |
| **Combined** | **1000** | **1239.82** | **23.98** | — |

### Transaction Fee Sensitivity

**Gold:**

| Fee (%) | Final Value ($) |
|---|---|
| 0.2 | 417.93 |
| 0.4 | 345.42 |
| 0.6 | 289.27 |
| 0.8 | 254.54 |
| 1.0 | 216.36 |
| 1.2 | 175.37 |

**Bitcoin:**

| Fee (%) | Final Value ($) |
|---|---|
| 1.5 | 821.89 |
| 2.0 | 559.31 |
| 2.5 | 376.99 |
| 3.0 | 246.49 |
| 3.5 | 155.74 |
