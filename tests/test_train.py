"""Tests for src/train.py."""

import os

import numpy as np
import tensorflow as tf

from src.model import build_model
from src.train import build_callbacks, prepare_for_training, set_seeds, train

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VOCAB = 20
_SEQ = 8
_BATCH = 4


def _small_model() -> tf.keras.Model:
    return build_model(vocab_size=_VOCAB, embed_dim=8, lstm_units=16, num_lstm_layers=1, dropout_rate=0.0)


def _tiny_dataset(n_samples: int = 32) -> tf.data.Dataset:
    """Synthetic (input, target_seq) dataset matching build_dataset output format."""
    inputs = tf.random.uniform((_BATCH, _SEQ), 0, _VOCAB, dtype=tf.int32)
    targets = tf.random.uniform((_BATCH, _SEQ), 0, _VOCAB, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensors((inputs, targets)).repeat(n_samples // _BATCH)
    return ds


# ---------------------------------------------------------------------------
# set_seeds
# ---------------------------------------------------------------------------


class TestSetSeeds:
    def test_runs_without_error(self):
        set_seeds(42)

    def test_different_seeds_give_different_results(self):
        set_seeds(1)
        a = np.random.rand()
        set_seeds(2)
        b = np.random.rand()
        assert a != b

    def test_same_seed_gives_same_numpy_result(self):
        set_seeds(99)
        a = np.random.rand(5)
        set_seeds(99)
        b = np.random.rand(5)
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# prepare_for_training
# ---------------------------------------------------------------------------


class TestPrepareForTraining:
    def test_target_shape_becomes_1d(self):
        ds = _tiny_dataset()
        ready = prepare_for_training(ds)
        _, targets = next(iter(ready))
        assert targets.shape == (_BATCH,)

    def test_input_shape_unchanged(self):
        ds = _tiny_dataset()
        ready = prepare_for_training(ds)
        inputs, _ = next(iter(ready))
        assert inputs.shape == (_BATCH, _SEQ)

    def test_target_is_last_element_of_sequence(self):
        inputs_data = tf.constant([[0, 1, 2, 3]], dtype=tf.int32)
        targets_data = tf.constant([[10, 11, 12, 99]], dtype=tf.int32)
        ds = tf.data.Dataset.from_tensors((inputs_data, targets_data))
        ready = prepare_for_training(ds)
        _, target = next(iter(ready))
        assert int(target[0]) == 99

    def test_returns_tf_dataset(self):
        assert isinstance(prepare_for_training(_tiny_dataset()), tf.data.Dataset)


# ---------------------------------------------------------------------------
# build_callbacks
# ---------------------------------------------------------------------------


class TestBuildCallbacks:
    def test_returns_three_callbacks(self, tmp_path):
        cbs = build_callbacks(str(tmp_path / "models"), str(tmp_path / "logs"), patience=3)
        assert len(cbs) == 3

    def test_has_model_checkpoint(self, tmp_path):
        cbs = build_callbacks(str(tmp_path / "models"), str(tmp_path / "logs"), patience=3)
        types = [type(c) for c in cbs]
        assert tf.keras.callbacks.ModelCheckpoint in types

    def test_has_early_stopping(self, tmp_path):
        cbs = build_callbacks(str(tmp_path / "models"), str(tmp_path / "logs"), patience=3)
        types = [type(c) for c in cbs]
        assert tf.keras.callbacks.EarlyStopping in types

    def test_has_tensorboard(self, tmp_path):
        cbs = build_callbacks(str(tmp_path / "models"), str(tmp_path / "logs"), patience=3)
        types = [type(c) for c in cbs]
        assert tf.keras.callbacks.TensorBoard in types

    def test_early_stopping_patience(self, tmp_path):
        cbs = build_callbacks(str(tmp_path / "models"), str(tmp_path / "logs"), patience=7)
        es = next(c for c in cbs if isinstance(c, tf.keras.callbacks.EarlyStopping))
        assert es.patience == 7

    def test_early_stopping_restores_best_weights(self, tmp_path):
        cbs = build_callbacks(str(tmp_path / "models"), str(tmp_path / "logs"), patience=3)
        es = next(c for c in cbs if isinstance(c, tf.keras.callbacks.EarlyStopping))
        assert es.restore_best_weights is True

    def test_checkpoint_monitors_val_loss(self, tmp_path):
        cbs = build_callbacks(str(tmp_path / "models"), str(tmp_path / "logs"), patience=3)
        ckpt = next(c for c in cbs if isinstance(c, tf.keras.callbacks.ModelCheckpoint))
        assert ckpt.monitor == "val_loss"

    def test_checkpoint_save_best_only(self, tmp_path):
        cbs = build_callbacks(str(tmp_path / "models"), str(tmp_path / "logs"), patience=3)
        ckpt = next(c for c in cbs if isinstance(c, tf.keras.callbacks.ModelCheckpoint))
        assert ckpt.save_best_only is True

    def test_model_dir_created(self, tmp_path):
        model_dir = str(tmp_path / "new_dir" / "models")
        build_callbacks(model_dir, str(tmp_path / "logs"), patience=3)
        assert os.path.isdir(model_dir)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


class TestTrain:
    def test_returns_history(self, tmp_path):
        model = _small_model()
        ds = _tiny_dataset()
        history = train(
            model, ds, ds,
            epochs=1, learning_rate=0.01,
            model_dir=str(tmp_path / "models"),
            log_dir=str(tmp_path / "logs"),
            patience=3,
            steps_per_epoch=2,
            validation_steps=1,
        )
        assert isinstance(history, tf.keras.callbacks.History)

    def test_history_contains_loss_keys(self, tmp_path):
        model = _small_model()
        ds = _tiny_dataset()
        history = train(
            model, ds, ds,
            epochs=1, learning_rate=0.01,
            model_dir=str(tmp_path / "models"),
            log_dir=str(tmp_path / "logs"),
            patience=3,
            steps_per_epoch=2,
            validation_steps=1,
        )
        assert "loss" in history.history
        assert "val_loss" in history.history
        assert "accuracy" in history.history

    def test_checkpoint_file_created(self, tmp_path):
        model = _small_model()
        ds = _tiny_dataset()
        model_dir = str(tmp_path / "models")
        train(
            model, ds, ds,
            epochs=1, learning_rate=0.01,
            model_dir=model_dir,
            log_dir=str(tmp_path / "logs"),
            patience=3,
            steps_per_epoch=2,
            validation_steps=1,
        )
        assert os.path.exists(os.path.join(model_dir, "best_model.keras"))

    def test_saved_model_is_loadable(self, tmp_path):
        model = _small_model()
        ds = _tiny_dataset()
        model_dir = str(tmp_path / "models")
        train(
            model, ds, ds,
            epochs=1, learning_rate=0.01,
            model_dir=model_dir,
            log_dir=str(tmp_path / "logs"),
            patience=3,
            steps_per_epoch=2,
            validation_steps=1,
        )
        loaded = tf.keras.models.load_model(os.path.join(model_dir, "best_model.keras"))
        assert isinstance(loaded, tf.keras.Model)

    def test_loaded_model_output_shape(self, tmp_path):
        model = _small_model()
        ds = _tiny_dataset()
        model_dir = str(tmp_path / "models")
        train(
            model, ds, ds,
            epochs=1, learning_rate=0.01,
            model_dir=model_dir,
            log_dir=str(tmp_path / "logs"),
            patience=3,
            steps_per_epoch=2,
            validation_steps=1,
        )
        loaded = tf.keras.models.load_model(os.path.join(model_dir, "best_model.keras"))
        out = loaded(tf.constant([[1, 2, 3]], dtype=tf.int32), training=False)
        assert out.shape == (1, _VOCAB)

    def test_loss_decreases_when_overfitting(self, tmp_path):
        """Model should be able to overfit a tiny repeated dataset."""
        set_seeds(0)
        model = _small_model()
        # Repeat the same batch many times so the model can memorise it
        inputs = tf.constant([[0, 1, 2, 3, 4, 5, 6, 7]] * _BATCH, dtype=tf.int32)
        targets = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8]] * _BATCH, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensors((inputs, targets)).repeat()

        history = train(
            model, ds, ds,
            epochs=15, learning_rate=0.05,
            model_dir=str(tmp_path / "models"),
            log_dir=str(tmp_path / "logs"),
            patience=20,  # disable early stopping for this test
            steps_per_epoch=4,
            validation_steps=1,
        )
        losses = history.history["loss"]
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_tensorboard_logs_created(self, tmp_path):
        model = _small_model()
        ds = _tiny_dataset()
        log_dir = str(tmp_path / "logs")
        train(
            model, ds, ds,
            epochs=1, learning_rate=0.01,
            model_dir=str(tmp_path / "models"),
            log_dir=log_dir,
            patience=3,
            steps_per_epoch=2,
            validation_steps=1,
        )
        fit_dir = os.path.join(log_dir, "fit")
        assert os.path.isdir(fit_dir)
        assert len(os.listdir(fit_dir)) > 0
