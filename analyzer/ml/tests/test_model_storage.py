"""Tests for durable DB-backed model-artifact storage (issue #91).

The point of the feature is that a model trained on one container is loadable on
another with no shared disk. These tests prove the artifact round-trips, that a
corrupted artifact is rejected rather than executed, and - the crux - that the
grade-path loader (`HybridQueryGrader._load_current_model`) resolves the model
from the database even when no local `.pkl` file exists.
"""

import os
import tempfile

from django.test import TestCase
from sklearn.linear_model import LinearRegression

from analyzer.ml.core import model_storage
from analyzer.ml.core.hybrid_grader import HybridQueryGrader
from analyzer.models import MLModel, MLModelArtifact


def _fitted_model():
    model = LinearRegression()
    model.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
    return model


def _make_row(model_type="HYBRID_SCORER", status="ACTIVE", version="t1"):
    return MLModel.objects.create(
        name="test model",
        model_type=model_type,
        version=version,
        status=status,
        file_path=f"{model_type}_{version}.pkl",
    )


class ModelStorageRoundTripTests(TestCase):
    def test_store_and_retrieve_returns_equivalent_model(self):
        row = _make_row()
        model = _fitted_model()

        model_storage.store(row, model)
        loaded = model_storage.retrieve(row)

        self.assertIsNotNone(loaded)
        # A LinearRegression that survived serialization predicts identically.
        self.assertAlmostEqual(loaded.predict([[3.0]])[0], 3.0, places=6)

    def test_retrieve_returns_none_when_no_artifact(self):
        row = _make_row()
        self.assertIsNone(model_storage.retrieve(row))

    def test_store_replaces_existing_artifact(self):
        row = _make_row()
        model_storage.store(row, _fitted_model())
        model_storage.store(row, {"replaced": True})

        # OneToOne: exactly one artifact row, holding the newer object.
        self.assertEqual(MLModelArtifact.objects.filter(model=row).count(), 1)
        self.assertEqual(model_storage.retrieve(row), {"replaced": True})

    def test_checksum_mismatch_is_rejected(self):
        row = _make_row()
        model_storage.store(row, _fitted_model())

        artifact = MLModelArtifact.objects.get(model=row)
        artifact.data = bytes(artifact.data) + b"tampered"
        artifact.save(update_fields=["data"])

        # The stored checksum no longer matches; retrieve refuses to unpickle.
        self.assertIsNone(model_storage.retrieve(row))

    def test_deleting_model_cascades_to_artifact(self):
        row = _make_row()
        model_storage.store(row, _fitted_model())
        self.assertTrue(MLModelArtifact.objects.exists())

        row.delete()
        self.assertFalse(MLModelArtifact.objects.exists())


class GradePathLoadsFromDatabaseTests(TestCase):
    """The #91 scenario: load the active model with no local file at all."""

    def test_load_current_model_uses_db_artifact_without_local_file(self):
        row = _make_row()
        bundle = {
            "model": _fitted_model(),
            "scaler": None,
            "feature_names": ["f0"],
        }
        model_storage.store(row, bundle)

        grader = HybridQueryGrader()
        # Point the legacy file dir at an empty temp dir so the ONLY way to
        # load is the DB artifact. If the loader still returned a model, it
        # came from the database.
        with tempfile.TemporaryDirectory() as empty_dir:
            grader.model_path = empty_dir
            self.assertFalse(os.path.exists(os.path.join(empty_dir, row.file_path)))
            loaded = grader._load_current_model()

        self.assertIsNotNone(loaded)
        self.assertAlmostEqual(loaded.predict([[4.0]])[0], 4.0, places=6)

    def test_load_current_model_none_when_no_artifact_and_no_file(self):
        _make_row()  # active row, but no artifact stored and no file on disk
        grader = HybridQueryGrader()
        with tempfile.TemporaryDirectory() as empty_dir:
            grader.model_path = empty_dir
            self.assertIsNone(grader._load_current_model())
