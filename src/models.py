"""
Model implementations for price prediction benchmarking.

Models:
  1. Att-BiLSTM  — Attention-augmented Bidirectional LSTM (paper model)
  2. LSTM        — Vanilla LSTM baseline
  3. GRU         — Gated Recurrent Unit baseline
  4. BiLSTM      — Bidirectional LSTM (no attention)
  5. XGBoost     — Gradient Boosted Trees
  6. ARIMA       — Auto-ARIMA
  7. HMM         — Gaussian Hidden Markov Model (hmmlearn)
  8. Holt-Winters — Triple Exponential Smoothing

# Using hmmlearn GaussianHMM — Context7 confirmed
# Using pmdarima auto_arima — Context7 confirmed
"""
import random
import warnings

import numpy as np

warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.random.set_seed(42)

from tensorflow.keras import layers
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from hmmlearn.hmm import GaussianHMM
from pmdarima import auto_arima


# ---------------------------------------------------------------------------
# Custom Attention Layer
# ---------------------------------------------------------------------------

class DotProductAttention(tf.keras.layers.Layer):
    """Dot-product attention with a trainable query vector.

    Computes attention scores S(x_i, q) = x_i^T × q for each timestep,
    applies softmax to get weights a_i, and returns the weighted sum
    context vector: att(q, x) = Σ_i a_i × x_i.

    # Custom — Context7 found no library equivalent (paper Section 3.3)
    """

    def build(self, input_shape: tuple) -> None:
        # input_shape: (batch, timesteps, features)
        self.query = self.add_weight(
            name="query",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # scores: (batch, timesteps)
        scores = tf.matmul(x, tf.expand_dims(self.query, -1))
        scores = tf.squeeze(scores, -1)
        weights = tf.nn.softmax(scores, axis=-1)
        weights = tf.expand_dims(weights, -1)
        context = tf.reduce_sum(x * weights, axis=1)
        return context

    def get_config(self) -> dict:
        return super().get_config()


# ---------------------------------------------------------------------------
# Model Builders
# ---------------------------------------------------------------------------

def build_att_bilstm(window: int = 100) -> tf.keras.Model:
    """Build Att-BiLSTM model.

    Architecture (paper Section 3.3):
      Input(window, 1) → BiLSTM(128, return_seq=True) → Dropout(0.2)
      → DotProductAttention → Dropout(0.2) → Reshape → LSTM(64) → Dense(1)

    Args:
        window: Input sequence length (default 100).

    Returns:
        Compiled Keras model.
    """
    inputs = layers.Input(shape=(window, 1))
    # Paper: 128 units total = 64 forward + 64 backward
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    x = layers.Dropout(0.2)(x)
    x = DotProductAttention()(x)          # → (batch, 128)
    x = layers.Dropout(0.2)(x)
    x = layers.Reshape((1, 128))(x)       # needed for second LSTM
    x = layers.LSTM(64)(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs, name="Att_BiLSTM")
    model.compile(
        loss="mae",
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    )
    return model


def build_lstm(window: int = 100) -> tf.keras.Model:
    """Build vanilla LSTM baseline.

    Architecture: Input(window,1) → LSTM(128) → Dropout(0.2) → Dense(1)

    Args:
        window: Input sequence length (default 100).

    Returns:
        Compiled Keras model.
    """
    inputs = layers.Input(shape=(window, 1))
    x = layers.LSTM(128)(inputs)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs, name="LSTM")
    model.compile(
        loss="mae",
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    )
    return model


def build_gru(window: int = 100) -> tf.keras.Model:
    """Build GRU baseline.

    Architecture: Input(window,1) → GRU(128) → Dropout(0.2) → Dense(1)

    Args:
        window: Input sequence length (default 100).

    Returns:
        Compiled Keras model.
    """
    inputs = layers.Input(shape=(window, 1))
    x = layers.GRU(128)(inputs)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs, name="GRU")
    model.compile(
        loss="mae",
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    )
    return model


def build_bilstm(window: int = 100) -> tf.keras.Model:
    """Build Bidirectional LSTM (no attention) baseline.

    Architecture: Input(window,1) → Bidirectional(LSTM(128)) → Dropout(0.2) → Dense(1)

    Args:
        window: Input sequence length (default 100).

    Returns:
        Compiled Keras model.
    """
    inputs = layers.Input(shape=(window, 1))
    x = layers.Bidirectional(layers.LSTM(128))(inputs)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs, name="BiLSTM")
    model.compile(
        loss="mae",
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    )
    return model


# ---------------------------------------------------------------------------
# Training Utility
# ---------------------------------------------------------------------------

def train_dl_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    val_split: float = 0.1,
    patience: int = 10,
) -> tf.keras.callbacks.History:
    """Train a Keras model with early stopping.

    Uses 10% of training data for validation; restores best weights on stop.

    Args:
        model: Compiled Keras model.
        X_train: Training inputs of shape (N, window, 1).
        y_train: Training targets of shape (N,).
        epochs: Maximum epochs (default 100).
        batch_size: Mini-batch size (default 32).
        val_split: Fraction of training data for validation (default 0.1).
        patience: Early-stopping patience (default 10).

    Returns:
        Training history object.
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[early_stop],
        verbose=0,
    )
    return history


# ---------------------------------------------------------------------------
# Benchmark Models
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Train XGBoost regressor on flattened sliding windows.

    Params: n_estimators=500, max_depth=5, learning_rate=0.05.

    Args:
        X_train: Training windows of shape (N_train, window, 1).
        y_train: Training targets of shape (N_train,).
        X_test: Test windows of shape (N_test, window, 1).

    Returns:
        Predictions array of shape (N_test,) in normalised scale.
    """
    X_tr_flat = X_train.reshape(X_train.shape[0], -1)
    X_te_flat = X_test.reshape(X_test.shape[0], -1)
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_tr_flat, y_train)
    return xgb.predict(X_te_flat)


def train_arima(train_prices: np.ndarray, test_len: int) -> np.ndarray:
    """Fit auto-ARIMA on raw training prices and forecast test period.

    # Using pmdarima auto_arima — Context7 confirmed

    Args:
        train_prices: 1-D array of raw (non-normalised) training prices.
        test_len: Number of steps to forecast.

    Returns:
        Forecast array of shape (test_len,).
    """
    model = auto_arima(
        train_prices,
        stepwise=True,
        information_criterion="aic",
        error_action="ignore",
        suppress_warnings=True,
        max_p=5,
        max_q=5,
    )
    forecasts = model.predict(n_periods=test_len)
    return np.array(forecasts, dtype=float)


def train_hmm_predict(
    train_M: np.ndarray,
    train_prices: np.ndarray,
    test_len: int,
) -> np.ndarray:
    """Fit GaussianHMM on training M_i and predict test prices iteratively.

    # Using hmmlearn GaussianHMM — Context7 confirmed
    Strategy:
      1. Fit 4-state Gaussian HMM on training M_i values.
      2. Obtain Viterbi state sequence for training data.
      3. Compute mean M_i per hidden state from training data.
      4. Iteratively predict next price by expected M_i from state transitions.
         Price update: p_next = p_current * (1 + E[M])

    Args:
        train_M: 1-D array of training M_i values.
        train_prices: 1-D array of raw training prices (used for the last price).
        test_len: Number of steps to predict.

    Returns:
        Predicted price array of shape (test_len,).
    """
    hmm_model = GaussianHMM(
        n_components=4,
        covariance_type="full",
        n_iter=100,
        random_state=42,
    )
    train_M_2d = train_M.reshape(-1, 1)
    hmm_model.fit(train_M_2d)

    train_states = hmm_model.predict(train_M_2d)

    # Mean M_i per state from training
    state_means = np.array(
        [
            train_M[train_states == s].mean() if (train_states == s).sum() > 0 else 0.0
            for s in range(hmm_model.n_components)
        ]
    )
    trans_mat = hmm_model.transmat_

    current_state = int(train_states[-1])
    current_price = float(train_prices[-1])

    predictions = []
    for _ in range(test_len):
        next_probs = trans_mat[current_state]
        expected_M = float(np.dot(next_probs, state_means))
        # p_i = p_{i-1} * (1 + M_i)  (approximate for small M_i)
        next_price = current_price * (1.0 + expected_M)
        predictions.append(next_price)
        current_state = int(np.argmax(next_probs))
        current_price = next_price

    return np.array(predictions, dtype=float)


def train_holtwinters(train_prices: np.ndarray, test_len: int) -> np.ndarray:
    """Fit Holt-Winters (additive trend, no seasonality) and forecast.

    # Using statsmodels ExponentialSmoothing — Context7 confirmed

    Args:
        train_prices: 1-D array of raw training prices.
        test_len: Number of steps to forecast.

    Returns:
        Forecast array of shape (test_len,).
    """
    model = ExponentialSmoothing(train_prices, trend="add", seasonal=None)
    fitted = model.fit(optimized=True)
    forecasts = fitted.forecast(test_len)
    return np.array(forecasts, dtype=float)


# ---------------------------------------------------------------------------
# Multi-step Forecast Generation
# ---------------------------------------------------------------------------

def generate_multistep_forecasts(
    model: tf.keras.Model,
    prices_norm: np.ndarray,
    train_end_idx: int,
    window: int = 100,
    steps: int = 3,
) -> np.ndarray:
    """Generate iterative multi-step ahead forecasts for each test day.

    For each test day t (from train_end_idx to end), the model predicts
    the next `steps` normalised prices using iterative one-step prediction
    (each new prediction is appended to the input window for the next step).

    Args:
        model: Trained Keras model expecting input (1, window, 1).
        prices_norm: Full normalised price series (train + test).
        train_end_idx: Index where test period begins (inclusive).
        window: Sliding window length (default 100).
        steps: Number of steps ahead per test day (default 3).

    Returns:
        Array of shape (n_test, steps) with normalised forecasts.
    """
    n = len(prices_norm)
    forecasts = []

    for t in range(train_end_idx, n):
        current_seq = list(prices_norm[max(0, t - window + 1) : t + 1])
        # Pad if not enough history (only happens at very beginning)
        while len(current_seq) < window:
            current_seq.insert(0, current_seq[0])

        step_forecasts = []
        for _ in range(steps):
            inp = np.array(current_seq[-window:], dtype=float).reshape(1, window, 1)
            pred = float(model.predict(inp, verbose=0)[0][0])
            step_forecasts.append(pred)
            current_seq.append(pred)

        forecasts.append(step_forecasts)

    return np.array(forecasts, dtype=float)
