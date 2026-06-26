# Repository Knowledge — Att-BiLSTM Price Prediction

## Project Structure

```
src/                    ← core implementation
exp/                    ← experiment scripts (call src functions)
tests/                  ← pytest unit + integration tests
data/                   ← cached Yahoo Finance CSVs
results/                ← plots, tables, RESULTS.md
```

## Key Source Modules

| Module | Purpose |
|--------|---------|
| `src/data_loader.py` | `get_prices(ticker)`, `compute_M_series()`, `normalize_prices()`, `create_sliding_windows()`, `train_test_split_temporal()` |
| `src/models.py` | `build_att_bilstm()`, `build_lstm()`, `build_gru()`, `build_bilstm()`, `train_dl_model()`, `train_xgboost()`, `train_arima()`, `train_hmm_predict()`, `train_holtwinters()`, `generate_multistep_forecasts()`, `DotProductAttention` |
| `src/evaluation.py` | `rmse()`, `mae()`, `mape()`, `r_squared()`, `compute_all_metrics()` |
| `src/apriori_analysis.py` | `compute_runs()`, `get_run_counts()`, `compute_u_n()`, `compute_percentile_stats()`, `validate_run_counts()`, `full_apriori_analysis()` |
| `src/position_sizing.py` | `derive_position_sizes()`, `fit_exponential()`, `compute_allocations()`, `full_position_sizing()` |
| `src/trading_strategy.py` | `compute_var()`, `generate_signals()`, `simulate_trading()`, `run_sensitivity_analysis()` |

## Critical Formula Details

- **M_i formula**: `M[i] = (price[i] - price[i-1]) / price[i]`  ← denominator is CURRENT price
- **Sliding window**: length=100, step=1; target = price at t+window
- **Train/test split**: 70%/30% chronological on window array; training windows are randomly shuffled
- **Normalization**: MinMaxScaler fitted on first 70% of raw price series only
- **Att-BiLSTM**: BiLSTM(128) → DotProductAttention → LSTM(64) → Dense(1); compiled with MAE loss, RMSprop(lr=0.001)
- **VaR**: 95% confidence, 100-day rolling historical percentile of M_i
- **Position sizing break-even**: `p_n = -(1/M_u05) × Σ_{k=1}^{n-1} p_k × M_u05^{n-1-k} × (M_d01/u_n)^{n-k}`

## Data Details

- Tickers: `GC=F` (Gold), `BTC-USD` (Bitcoin)
- Date range: 2015-01-01 to 2021-09-10
- Cached CSVs: `data/GCF_prices.csv`, `data/BTC_USD_prices.csv`
- Gold: ~1,600 trading days (weekdays only); Bitcoin: ~2,400 calendar days

## Dependencies

Python 3.12, TensorFlow 2.21, Keras 3, pmdarima, hmmlearn, statsmodels, xgboost, yfinance, scikit-learn, numpy, pandas, matplotlib, scipy.

## Test Patterns

- All tests use synthetic data (no network calls)
- Run with: `pytest tests/ -v`
- Tests are pure unit tests — no mocks (real code paths only)
- Deep learning tests use WINDOW=10, tiny data to keep runtime fast

## Running Experiments

```bash
python exp/run_all.py      # all 4 experiments in sequence
python exp/run_exp2.py     # individual experiment
```

Exp results go to `results/` directory.
Exp chain: exp2 → exp3 (uses exp2 params) → exp4 (uses exp3 params + exp1 predictions).

## Known Issues / Notes

- `yfinance` column output may be a MultiIndex DataFrame; `get_prices()` handles this with `.squeeze()`.
- `generate_multistep_forecasts()` does iterative 3-step rollout; on very short sequences pads with first value.
- HMM uses Viterbi state sequence + state transition for iterative price prediction.
- For exp4 standalone run (without exp1), a simple 3-day rolling mean is used as fallback forecast.
