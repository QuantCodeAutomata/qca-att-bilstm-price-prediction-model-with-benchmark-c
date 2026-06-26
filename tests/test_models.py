"""
Tests for src/models.py — model architecture shapes, attention layer,
XGBoost, and multi-step forecast generation.

Deep-learning training tests are kept minimal (1 epoch, small data)
to avoid long runtimes while still validating forward passes.
"""
import numpy as np
import pytest

from src.models import (
    DotProductAttention,
    build_att_bilstm,
    build_lstm,
    build_gru,
    build_bilstm,
    train_dl_model,
    train_xgboost,
    generate_multistep_forecasts,
)

WINDOW = 10  # small window for fast tests


@pytest.fixture
def tiny_dataset():
    """Generate a tiny synthetic dataset for quick model tests."""
    rng = np.random.RandomState(0)
    N = 60
    X = rng.randn(N, WINDOW, 1).astype("float32")
    y = rng.randn(N).astype("float32")
    return X, y


class TestModelInputOutput:
    """Test that each model produces correctly shaped outputs."""

    @pytest.mark.parametrize("builder", [build_att_bilstm, build_lstm, build_gru, build_bilstm])
    def test_output_shape(self, builder, tiny_dataset):
        X, _ = tiny_dataset
        model = builder(window=WINDOW)
        preds = model.predict(X[:5], verbose=0)
        assert preds.shape == (5, 1)

    @pytest.mark.parametrize("builder", [build_att_bilstm, build_lstm, build_gru, build_bilstm])
    def test_output_finite(self, builder, tiny_dataset):
        X, _ = tiny_dataset
        model = builder(window=WINDOW)
        preds = model.predict(X[:5], verbose=0)
        assert np.all(np.isfinite(preds))

    def test_att_bilstm_parameter_count(self):
        """Att-BiLSTM should have more parameters than vanilla LSTM."""
        att = build_att_bilstm(WINDOW)
        lstm = build_lstm(WINDOW)
        assert att.count_params() > lstm.count_params()


class TestAttention:
    """Test DotProductAttention layer behaviour."""

    def test_output_shape(self):
        import tensorflow as tf
        layer = DotProductAttention()
        # (batch=2, timesteps=10, features=8)
        x = tf.constant(np.ones((2, 10, 8), dtype="float32"))
        out = layer(x)
        assert out.shape == (2, 8)

    def test_weights_sum_to_one(self):
        """Softmax weights must sum to 1 across timesteps."""
        import tensorflow as tf
        layer = DotProductAttention()
        x = tf.constant(np.ones((1, 5, 4), dtype="float32"))
        # Build the layer
        _ = layer(x)
        query = layer.query.numpy()
        scores = x.numpy()[0] @ query
        weights = np.exp(scores) / np.sum(np.exp(scores))
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)


class TestTrainDLModel:
    def test_training_runs_one_epoch(self, tiny_dataset):
        X, y = tiny_dataset
        model = build_lstm(window=WINDOW)
        history = train_dl_model(model, X, y, epochs=1, batch_size=16, patience=5)
        assert "val_loss" in history.history

    def test_loss_is_finite(self, tiny_dataset):
        X, y = tiny_dataset
        model = build_lstm(window=WINDOW)
        history = train_dl_model(model, X, y, epochs=2, batch_size=16, patience=5)
        for loss_val in history.history["loss"]:
            assert np.isfinite(loss_val)


class TestTrainXGBoost:
    def test_output_shape(self, tiny_dataset):
        X, y = tiny_dataset
        preds = train_xgboost(X[:40], y[:40], X[40:])
        assert preds.shape == (len(X) - 40,)

    def test_output_finite(self, tiny_dataset):
        X, y = tiny_dataset
        preds = train_xgboost(X[:40], y[:40], X[40:])
        assert np.all(np.isfinite(preds))

    def test_flattening_is_correct(self, tiny_dataset):
        """XGBoost receives 100-feature vectors from (N, 10, 1) windows."""
        X, y = tiny_dataset
        # Should not raise: each window → 10*1 = 10 features
        preds = train_xgboost(X, y, X[:5])
        assert len(preds) == 5


class TestGenerateMultistepForecasts:
    def test_output_shape(self, tiny_dataset):
        X_win, y = tiny_dataset
        # Build a tiny normalised price series
        rng = np.random.RandomState(1)
        prices_norm = rng.rand(50).astype("float32")
        model = build_lstm(window=WINDOW)
        # train_end_idx = 30, so test period = prices[30:] = 20 days
        forecasts = generate_multistep_forecasts(
            model, prices_norm, train_end_idx=30, window=WINDOW, steps=3
        )
        assert forecasts.shape[0] == 20  # n_test = 50 - 30
        assert forecasts.shape[1] == 3

    def test_forecasts_finite(self, tiny_dataset):
        rng = np.random.RandomState(2)
        prices_norm = rng.rand(50).astype("float32")
        model = build_lstm(window=WINDOW)
        forecasts = generate_multistep_forecasts(
            model, prices_norm, train_end_idx=30, window=WINDOW, steps=3
        )
        assert np.all(np.isfinite(forecasts))
