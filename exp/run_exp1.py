"""
Experiment 1: Att-BiLSTM Price Prediction Model with Benchmark Comparison.

Trains Att-BiLSTM + 7 benchmark models on gold and Bitcoin price data,
computes RMSE/MAE/MAPE/R² on the test set, and generates prediction plots.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import (
    get_prices,
    compute_M_series,
    normalize_prices,
    create_sliding_windows,
    train_test_split_temporal,
)
from src.models import (
    build_att_bilstm,
    build_lstm,
    build_gru,
    build_bilstm,
    train_dl_model,
    train_xgboost,
    train_arima,
    train_hmm_predict,
    train_holtwinters,
    generate_multistep_forecasts,
)
from src.evaluation import compute_all_metrics

RESULTS_DIR = "results"
WINDOW = 100
EPOCHS = 100
BATCH = 32
TRAIN_FRAC = 0.7


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #

def _plot_all_predictions(
    y_true: np.ndarray,
    predictions: dict,
    asset_name: str,
    suffix: str = "",
) -> str:
    """Plot predicted vs actual prices for all models."""
    n_models = len(predictions)
    cols = 2
    rows = (n_models + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()

    for idx, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]
        ax.plot(y_true, label="Actual", color="blue", lw=1.0)
        ax.plot(y_pred, label=name, color="red", lw=1.0, alpha=0.8)
        ax.set_title(f"{asset_name} — {name}")
        ax.set_xlabel("Test Day Index")
        ax.set_ylabel("Price ($)")
        ax.legend(fontsize=7)

    # Hide any unused subplots
    for idx in range(len(predictions), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"{asset_name} — Predicted vs Actual Prices", fontsize=14)
    plt.tight_layout()
    fpath = os.path.join(RESULTS_DIR, f"exp1_predictions_{suffix}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    return fpath


def _plot_btc_last200(
    y_true: np.ndarray,
    y_att_bilstm: np.ndarray,
) -> str:
    """Plot last 200 Bitcoin test days: actual (blue) vs Att-BiLSTM (red)."""
    n = 200
    y_t = y_true[-n:]
    y_p = y_att_bilstm[-n:]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(n), y_t, label="Actual", color="blue", lw=1.2)
    ax.plot(range(n), y_p, label="Att-BiLSTM Predicted", color="red", lw=1.2)
    ax.set_title("Bitcoin — Last 200 Test Days: Actual vs Att-BiLSTM Predicted")
    ax.set_xlabel("Test Day Index (last 200)")
    ax.set_ylabel("Bitcoin Price ($)")
    ax.legend()
    plt.tight_layout()
    fpath = os.path.join(RESULTS_DIR, "exp1_btc_last200.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    return fpath


def _save_metrics_table(metrics_all: dict, asset_name: str) -> pd.DataFrame:
    """Save metrics comparison table for all models on one asset."""
    rows = []
    for model_name, m in metrics_all.items():
        rows.append({
            "Model": model_name,
            "RMSE": round(m["RMSE"], 4),
            "MAE": round(m["MAE"], 4),
            "MAPE (%)": round(m["MAPE"], 4),
            "R²": round(m["R2"], 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(RESULTS_DIR, f"exp1_metrics_{asset_name.lower()}.csv"), index=False
    )
    return df


# --------------------------------------------------------------------------- #
# Core pipeline for a single asset
# --------------------------------------------------------------------------- #

def run_asset(asset_name: str, ticker: str) -> dict:
    """Train all 8 models on one asset and collect metrics + predictions.

    Args:
        asset_name: Human-readable name (e.g. 'Gold').
        ticker: Yahoo Finance ticker.

    Returns:
        Dict with 'metrics', 'predictions', 'y_true', 'att_bilstm_model',
        'scaler', 'prices', 'prices_norm', 'split_idx'.
    """
    print(f"\n  ====== {asset_name} ({ticker}) ======")

    # 1. Data
    prices = get_prices(ticker)
    print(f"  Downloaded {len(prices)} observations.")

    M_series = compute_M_series(prices)
    prices_norm, scaler = normalize_prices(prices, train_frac=TRAIN_FRAC)

    X, y = create_sliding_windows(prices_norm, window=WINDOW)
    X_train, X_test, y_train, y_test, split_idx = train_test_split_temporal(
        X, y, train_frac=TRAIN_FRAC, shuffle_train=True, seed=42
    )

    # Price split for classical models (align with window offset)
    # The first window ends at index WINDOW, so raw price split is WINDOW + split_idx
    raw_split = WINDOW + split_idx
    train_prices_raw = prices.values[:raw_split].astype(float)
    test_prices_raw = prices.values[raw_split:raw_split + len(y_test)].astype(float)
    train_M = M_series.values[:raw_split - 1].astype(float)

    print(f"  Train windows: {len(X_train)},  Test windows: {len(X_test)}")

    predictions = {}
    metrics = {}

    # ----------------------------------------------------------------------- #
    # DL models (Att-BiLSTM, LSTM, GRU, BiLSTM)
    # ----------------------------------------------------------------------- #
    dl_builders = {
        "Att-BiLSTM": build_att_bilstm,
        "LSTM": build_lstm,
        "GRU": build_gru,
        "BiLSTM": build_bilstm,
    }

    att_bilstm_model = None
    for model_name, builder_fn in dl_builders.items():
        print(f"  Training {model_name}...")
        model = builder_fn(window=WINDOW)
        train_dl_model(model, X_train, y_train, epochs=EPOCHS, batch_size=BATCH)
        y_pred_norm = model.predict(X_test, verbose=0).flatten()
        y_pred_raw = scaler.inverse_transform(
            y_pred_norm.reshape(-1, 1)
        ).flatten()
        predictions[model_name] = y_pred_raw
        metrics[model_name] = compute_all_metrics(test_prices_raw, y_pred_raw)
        print(f"    {model_name}: RMSE={metrics[model_name]['RMSE']:.2f}, "
              f"R²={metrics[model_name]['R2']:.4f}")
        if model_name == "Att-BiLSTM":
            att_bilstm_model = model

    # ----------------------------------------------------------------------- #
    # XGBoost
    # ----------------------------------------------------------------------- #
    print("  Training XGBoost...")
    y_pred_xgb_norm = train_xgboost(X_train, y_train, X_test)
    y_pred_xgb_raw = scaler.inverse_transform(
        y_pred_xgb_norm.reshape(-1, 1)
    ).flatten()
    predictions["XGBoost"] = y_pred_xgb_raw
    metrics["XGBoost"] = compute_all_metrics(test_prices_raw, y_pred_xgb_raw)
    print(f"    XGBoost: RMSE={metrics['XGBoost']['RMSE']:.2f}, "
          f"R²={metrics['XGBoost']['R2']:.4f}")

    # ----------------------------------------------------------------------- #
    # ARIMA
    # ----------------------------------------------------------------------- #
    print("  Training ARIMA (auto)...")
    try:
        y_pred_arima = train_arima(train_prices_raw, len(test_prices_raw))
        # Clip to positive range
        y_pred_arima = np.clip(y_pred_arima, 0, None)
    except Exception as e:
        print(f"    ARIMA failed ({e}), using naive forecast.")
        y_pred_arima = np.full(len(test_prices_raw), train_prices_raw[-1])
    predictions["ARIMA"] = y_pred_arima
    metrics["ARIMA"] = compute_all_metrics(test_prices_raw, y_pred_arima)
    print(f"    ARIMA: RMSE={metrics['ARIMA']['RMSE']:.2f}, "
          f"R²={metrics['ARIMA']['R2']:.4f}")

    # ----------------------------------------------------------------------- #
    # HMM
    # ----------------------------------------------------------------------- #
    print("  Training HMM...")
    try:
        y_pred_hmm = train_hmm_predict(train_M, train_prices_raw, len(test_prices_raw))
    except Exception as e:
        print(f"    HMM failed ({e}), using naive forecast.")
        y_pred_hmm = np.full(len(test_prices_raw), train_prices_raw[-1])
    predictions["HMM"] = y_pred_hmm
    metrics["HMM"] = compute_all_metrics(test_prices_raw, y_pred_hmm)
    print(f"    HMM: RMSE={metrics['HMM']['RMSE']:.2f}, "
          f"R²={metrics['HMM']['R2']:.4f}")

    # ----------------------------------------------------------------------- #
    # Holt-Winters
    # ----------------------------------------------------------------------- #
    print("  Training Holt-Winters...")
    try:
        y_pred_hw = train_holtwinters(train_prices_raw, len(test_prices_raw))
    except Exception as e:
        print(f"    Holt-Winters failed ({e}), using linear trend.")
        # Fallback: linear extrapolation
        x = np.arange(len(train_prices_raw))
        coeffs = np.polyfit(x[-50:], train_prices_raw[-50:], 1)
        y_pred_hw = np.polyval(
            coeffs, np.arange(len(train_prices_raw), len(train_prices_raw) + len(test_prices_raw))
        )
    predictions["Holt-Winters"] = y_pred_hw
    metrics["Holt-Winters"] = compute_all_metrics(test_prices_raw, y_pred_hw)
    print(f"    Holt-Winters: RMSE={metrics['Holt-Winters']['RMSE']:.2f}, "
          f"R²={metrics['Holt-Winters']['R2']:.4f}")

    # ----------------------------------------------------------------------- #
    # Generate 3-day forecasts for trading strategy (exp_4)
    # ----------------------------------------------------------------------- #
    print("  Generating 3-day ahead forecasts for exp_4...")
    train_end_norm_idx = WINDOW + split_idx
    forecasts_3day = generate_multistep_forecasts(
        att_bilstm_model,
        prices_norm,
        train_end_idx=train_end_norm_idx,
        window=WINDOW,
        steps=3,
    )
    # Inverse-transform multi-step forecasts
    forecasts_3day_raw = scaler.inverse_transform(
        forecasts_3day.reshape(-1, 1)
    ).reshape(forecasts_3day.shape)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "y_true": test_prices_raw,
        "att_bilstm_model": att_bilstm_model,
        "scaler": scaler,
        "prices": prices,
        "prices_norm": prices_norm,
        "split_idx": split_idx,
        "M_series": M_series,
        "forecasts_3day_raw": forecasts_3day_raw,
        "test_prices": pd.Series(
            test_prices_raw,
            index=prices.index[raw_split : raw_split + len(y_test)],
        ),
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def run_exp1() -> dict:
    """Run Experiment 1: train all 8 models, evaluate, and save results.

    Returns:
        Dictionary with results for 'gold' and 'btc'.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    gold_results = run_asset("Gold", "GC=F")
    btc_results = run_asset("Bitcoin", "BTC-USD")

    # ------------------------------------------------------------------ #
    # Metrics tables
    # ------------------------------------------------------------------ #
    gold_df = _save_metrics_table(gold_results["metrics"], "gold")
    btc_df = _save_metrics_table(btc_results["metrics"], "btc")

    print("\n  ---- Gold Metrics ----")
    print(gold_df.to_string(index=False))
    print("\n  ---- Bitcoin Metrics ----")
    print(btc_df.to_string(index=False))

    # Combined table
    gold_df["Asset"] = "Gold"
    btc_df["Asset"] = "Bitcoin"
    combined = pd.concat([gold_df, btc_df], ignore_index=True)
    combined.to_csv(
        os.path.join(RESULTS_DIR, "exp1_metrics_combined.csv"), index=False
    )

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    print("  Generating prediction plots...")
    _plot_all_predictions(
        gold_results["y_true"],
        gold_results["predictions"],
        "Gold",
        suffix="gold",
    )
    _plot_all_predictions(
        btc_results["y_true"],
        btc_results["predictions"],
        "Bitcoin",
        suffix="btc",
    )
    _plot_btc_last200(
        btc_results["y_true"],
        btc_results["predictions"]["Att-BiLSTM"],
    )

    print("  Experiment 1 complete. Results saved to results/exp1_*")

    return {"gold": gold_results, "btc": btc_results}


if __name__ == "__main__":
    run_exp1()
