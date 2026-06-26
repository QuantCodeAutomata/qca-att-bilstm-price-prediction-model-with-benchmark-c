# Att-BiLSTM Price Prediction & Quantitative Trading Strategy

A complete quantitative finance research repository implementing an
Attention-augmented Bidirectional LSTM (Att-BiLSTM) model for gold and
Bitcoin price prediction, combined with an Apriori-inspired position-sizing
model and a full trading strategy simulation.

## Research Overview

This project reproduces and extends the methodology from a quantitative
trading paper that proposes:

1. **Att-BiLSTM** — dot-product attention over a Bidirectional LSTM for
   next-day price prediction.
2. **Apriori run-length analysis** — statistical characterisation of
   consecutive rise/fall patterns in the rise/fall degree series *M_i*.
3. **Exponential position sizing** — break-even-derived position sizes fitted
   to *s_i = a · e^(b·i)*, providing disciplined capital allocation.
4. **Modified greedy trading algorithm** — combines BiLSTM signals, VaR
   filtering, and exponential position execution.

## Repository Structure

```
.
├── data/                   # Cached price CSVs + download script
├── src/                    # Core implementation modules
│   ├── data_loader.py      # yfinance download, normalisation, windowing
│   ├── models.py           # Att-BiLSTM, LSTM, GRU, BiLSTM, XGBoost, ARIMA, HMM
│   ├── evaluation.py       # RMSE, MAE, MAPE, R² metrics
│   ├── apriori_analysis.py # Run-length analysis, u_n, M percentiles
│   ├── position_sizing.py  # Break-even derivation, exponential fit, allocations
│   └── trading_strategy.py # VaR, signal generation, full simulation
├── exp/                    # Experiment scripts
│   ├── run_exp1.py         # Model training + benchmark comparison
│   ├── run_exp2.py         # Apriori pattern analysis
│   ├── run_exp3.py         # Position sizing derivation
│   ├── run_exp4.py         # Full trading simulation
│   └── run_all.py          # Run all experiments in sequence
├── tests/                  # pytest test suite
│   ├── test_evaluation.py
│   ├── test_data_loader.py
│   ├── test_apriori.py
│   ├── test_position_sizing.py
│   ├── test_models.py
│   └── test_trading_strategy.py
├── results/                # Generated plots, tables, RESULTS.md
│   └── RESULTS.md
├── requirements.txt
└── README.md
```

## Experiments

| ID   | Title | Key Output |
|------|-------|------------|
| exp_1 | Att-BiLSTM vs 7 benchmarks | RMSE/MAE/MAPE/R² comparison table |
| exp_2 | Apriori run-length analysis | u_n, M_u0.9, M_u0.5, M_d0.1 |
| exp_3 | Exponential position sizing | Allocation arrays y_i |
| exp_4 | Full trading simulation | Portfolio value curves, fee sensitivity |

## Benchmark Models

| Model | Library |
|-------|---------|
| Att-BiLSTM | TensorFlow/Keras (custom attention) |
| LSTM | TensorFlow/Keras |
| GRU | TensorFlow/Keras |
| BiLSTM | TensorFlow/Keras |
| XGBoost | xgboost |
| ARIMA | pmdarima (auto_arima) |
| HMM | hmmlearn (GaussianHMM) |
| Holt-Winters | statsmodels (ExponentialSmoothing) |

## Data

Daily closing prices downloaded from Yahoo Finance via `yfinance`:

| Asset   | Ticker  | Period |
|---------|---------|--------|
| Gold    | GC=F    | 2015-01-01 → 2021-09-10 |
| Bitcoin | BTC-USD | 2015-01-01 → 2021-09-10 |

Data is cached under `data/` after first download.

## Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Run a single experiment
python exp/run_exp2.py
python exp/run_exp3.py

# Run all experiments sequentially
python exp/run_all.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Results

See [`results/RESULTS.md`](results/RESULTS.md) for detailed metrics and findings.

### Target Performance (Att-BiLSTM)

| Asset   | RMSE    | MAE   | MAPE (%) | R²     |
|---------|---------|-------|----------|--------|
| Gold    | ~19.01  | ~13.98 | ~0.76   | ~0.939 |
| Bitcoin | ~1479   | ~906  | ~2.88    | ~0.993 |

### Trading Strategy Returns

| Asset   | Initial | Final      | Return     |
|---------|---------|------------|------------|
| Gold    | $500    | ~$646      | ~+29%      |
| Bitcoin | $500    | ~$215,487  | ~+43,097%  |
| **Total** | **$1,000** | **~$216,133** | **~+21,513%** |

## Methodology Notes

- **M_i formula**: `M_i = (p_i - p_{i-1}) / p_i` (current price as denominator)
- **Sliding window**: 100 timesteps, step=1
- **Train/test split**: 70%/30% chronological; only training set shuffled
- **Normalisation**: MinMaxScaler fitted on training data only
- **VaR**: 95% confidence, 100-day rolling historical window
