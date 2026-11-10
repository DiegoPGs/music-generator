"""Tests for src/train.py."""

import math
import os

import numpy as np
import pytest
import tensorflow as tf

from src.model import build_model
from src.train import (AcademicTrainingCallback, build_callbacks,
                       categorize_token_id, compute_layer_weight_stats,
                       compute_musical_domain_insights,
                       compute_sample_gradient_norms, describe_token,
                       inspect_pipeline_sample, prepare_for_training,
                       set_seeds, summarize_model_layers, train)

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
# Token Formatting & Categorization
# ---------------------------------------------------------------------------


class TestTokenDescriptionsAndCategories:
    def test_describe_note_on_includes_pitch_name(self):
        desc = describe_token(60)
        assert "NOTE_ON_60" in desc
        assert "C4" in desc

    def test_describe_note_off_includes_pitch_name(self):
        desc = describe_token(128 + 69)
        assert "NOTE_OFF_69" in desc
        assert "A4" in desc

    def test_describe_time_shift_includes_milliseconds(self):
        desc = describe_token(256 + 9)  # TIME_SHIFT_10
        assert "TIME_SHIFT_10" in desc
        assert "100ms" in desc

    def test_describe_set_velocity_includes_velocity_estimate(self):
        desc = describe_token(356)  # SET_VELOCITY_0
        assert "SET_VELOCITY_0" in desc
        assert "vel~" in desc

    def test_describe_unknown_token_fallback(self):
        desc = describe_token(9999, inv_vocab={})
        assert desc == "TOKEN_9999"

    def test_categorize_token_ids(self):
        assert categorize_token_id(0) == "NOTE_ON"
        assert categorize_token_id(128) == "NOTE_OFF"
        assert categorize_token_id(256) == "TIME_SHIFT"
        assert categorize_token_id(356) == "SET_VELOCITY"
        assert categorize_token_id(999, inv_vocab={}) == "OTHER"


# ---------------------------------------------------------------------------
# Academic Walkthrough & Diagnostic Helpers
# ---------------------------------------------------------------------------


class TestPipelineInspection:
    def test_inspect_pipeline_sample_metadata(self):
        raw_ds = _tiny_dataset()
        prep_ds = prepare_for_training(raw_ds)
        info = inspect_pipeline_sample(raw_ds, prep_ds, context_tokens=3)

        assert info is not None
        assert info["batch_size"] == _BATCH
        assert info["seq_len"] == _SEQ
        assert info["tokens_per_batch"] == _BATCH * _SEQ
        assert info["raw_input_shape"] == (_BATCH, _SEQ)
        assert info["raw_target_shape"] == (_BATCH, _SEQ)
        assert info["prepared_input_shape"] == (_BATCH, _SEQ)
        assert info["prepared_target_shape"] == (_BATCH,)
        assert len(info["sample_context_ids"]) == 3
        assert len(info["sample_context_descriptions"]) == 3
        assert isinstance(info["sample_target_id"], int)
        assert isinstance(info["sample_target_description"], str)

    def test_inspect_empty_dataset_returns_none(self):
        empty_ds = tf.data.Dataset.from_tensor_slices(
            (tf.zeros((0, _BATCH, _SEQ), dtype=tf.int32), tf.zeros((0, _BATCH, _SEQ), dtype=tf.int32))
        )
        prep_ds = prepare_for_training(empty_ds)
        assert inspect_pipeline_sample(empty_ds, prep_ds) is None


class TestModelDiagnostics:
    def test_summarize_model_layers(self):
        model = _small_model()
        layers_info = summarize_model_layers(model)
        assert len(layers_info) == len(model.layers)
        total_pct = sum(item["pct_params"] for item in layers_info)
        assert total_pct == pytest.approx(100.0, abs=1e-3)
        assert all("role" in item and item["role"] for item in layers_info)

    def test_compute_layer_weight_stats_tracks_updates(self):
        model = _small_model()
        initial_stats, snapshot = compute_layer_weight_stats(model)
        assert len(initial_stats) > 0
        assert all(item["update_l2"] == 0.0 for item in initial_stats)

        # Perturb first trainable weight and verify positive update norm
        model.trainable_weights[0].assign_add(tf.ones_like(model.trainable_weights[0]) * 0.1)
        updated_stats, _ = compute_layer_weight_stats(model, prev_weights=snapshot)
        assert updated_stats[0]["update_l2"] > 0.0
        assert updated_stats[0]["relative_update"] > 0.0

    def test_compute_sample_gradient_norms(self):
        model = _small_model()
        prep_ds = prepare_for_training(_tiny_dataset())
        sample_batch = next(iter(prep_ds))
        grad_norms = compute_sample_gradient_norms(model, sample_batch)

        assert "global" in grad_norms
        assert grad_norms["global"] > 0.0
        assert "embedding" in grad_norms
        assert "output" in grad_norms


class TestMusicalDomainInsights:
    def test_musical_domain_insights_structure_and_bounds(self):
        inv_vocab = {
            0: "NOTE_ON_60",
            1: "NOTE_OFF_60",
            2: "TIME_SHIFT_10",
            3: "SET_VELOCITY_16",
            **{i: f"NOTE_ON_{i}" for i in range(4, _VOCAB)},
        }
        model = _small_model()
        prep_ds = prepare_for_training(_tiny_dataset())
        sample_batch = next(iter(prep_ds))

        insights = compute_musical_domain_insights(model, sample_batch, inv_vocab=inv_vocab, top_k=3)
        pred_mass = sum(insights["predicted_category_probs"].values())
        target_share = sum(insights["target_category_shares"].values())

        assert pred_mass == pytest.approx(1.0, abs=1e-4)
        assert target_share == pytest.approx(1.0, abs=1e-4)
        assert 0.0 <= insights["mean_entropy"] <= insights["max_entropy"] + 1e-4
        assert insights["max_entropy"] == pytest.approx(math.log(_VOCAB), abs=1e-4)
        assert len(insights["sample_top_k"]) == 3
        assert 0.0 <= insights["mean_top1_confidence"] <= 1.0


# ---------------------------------------------------------------------------
# build_callbacks
# ---------------------------------------------------------------------------


class TestBuildCallbacks:
    def test_returns_four_callbacks(self, tmp_path):
        cbs = build_callbacks(str(tmp_path / "models"), str(tmp_path / "logs"), patience=3)
        assert len(cbs) == 4

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

    def test_has_academic_training_callback(self, tmp_path):
        cbs = build_callbacks(str(tmp_path / "models"), str(tmp_path / "logs"), patience=3)
        types = [type(c) for c in cbs]
        assert AcademicTrainingCallback in types

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
# AcademicTrainingCallback
# ---------------------------------------------------------------------------


class TestAcademicTrainingCallback:
    def test_records_epoch_diagnostics_during_fit(self):
        model = _small_model()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        prep_ds = prepare_for_training(_tiny_dataset())
        sample_batch = next(iter(prep_ds))

        cb = AcademicTrainingCallback(
            total_epochs=2,
            batch_size=_BATCH,
            seq_len=_SEQ,
            steps_per_epoch=2,
            patience=3,
            sample_batch=sample_batch,
            enable_progress_bar=False,
        )
        model.fit(prep_ds, validation_data=prep_ds, epochs=2, steps_per_epoch=2, validation_steps=1,
                  callbacks=[cb], verbose=0)

        assert len(cb.epoch_records) == 2
        first = cb.epoch_records[0]
        assert first["epoch"] == 1
        assert first["train_ppl"] == pytest.approx(math.exp(first["train_loss"]), rel=1e-4)
        assert first["val_ppl"] == pytest.approx(math.exp(first["val_loss"]), rel=1e-4)
        assert first["tokens_per_sec"] > 0.0
        assert len(first["weight_stats"]) > 0
        assert "global" in first["grad_norms"]
        assert first["musical_insights"] is not None

    def test_logs_all_four_academic_stages_cleanly(self, tmp_path, caplog):
        import logging

        model = _small_model()
        ds = _tiny_dataset()
        with caplog.at_level(logging.INFO, logger="src.train"):
            train(
                model,
                ds,
                ds,
                epochs=2,
                learning_rate=0.01,
                model_dir=str(tmp_path / "models"),
                log_dir=str(tmp_path / "logs"),
                patience=3,
                steps_per_epoch=2,
                validation_steps=1,
                enable_progress_bar=False,
            )

        assert "STAGE 1/4 — DATA & TENSOR PIPELINE WALKTHROUGH" in caplog.text
        assert "STAGE 2/4 — MODEL ARCHITECTURE & THEORETICAL BASELINES" in caplog.text
        assert "STAGE 3/4 — ITERATIVE OPTIMIZATION" in caplog.text
        assert "EPOCH 01/02 SUMMARY" in caplog.text
        assert "EPOCH 02/02 SUMMARY" in caplog.text
        assert "STAGE 4/4 — POST-TRAINING ACADEMIC SUMMARY" in caplog.text


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
            enable_progress_bar=False,
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
            enable_progress_bar=False,
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
            enable_progress_bar=False,
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
            enable_progress_bar=False,
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
            enable_progress_bar=False,
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
            enable_progress_bar=False,
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
            enable_progress_bar=False,
        )
        fit_dir = os.path.join(log_dir, "fit")
        assert os.path.isdir(fit_dir)
        assert len(os.listdir(fit_dir)) > 0
