"""Durable model-artifact storage backed by the shared database.

Trained models were written as ``.pkl`` files under a container-local
``ml_models`` directory. That disk is ephemeral on Railway and is not shared
between the web and worker services, so an ``MLModel.file_path`` row could point
at bytes that no longer exist, or that only ever existed on another container
(see issue #91). These helpers store the serialized model in an
``MLModelArtifact`` row instead, so any service that can read the ``MLModel``
row can also read its bytes.

All ML save/load sites route serialization through here, which also removes the
prior inconsistency where different modules stored a bare filename vs. a full
path in ``file_path`` and then joined it against three different directories.

The artifact is joblib-serialized bytes; ``store`` and ``retrieve`` deal in the
in-memory model object, so callers never touch the filesystem.
"""

import hashlib
import io
import logging
from typing import Any, Optional

import joblib

from ...models import MLModel, MLModelArtifact

logger = logging.getLogger(__name__)


def _serialize(obj: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return buffer.getvalue()


def _deserialize(data: bytes) -> Any:
    # joblib.load unpickles, which is arbitrary-code-execution on untrusted
    # input. These bytes are only ever produced by our own training pipeline
    # (store() below) and written to our own database; they never originate
    # from user input. retrieve() additionally verifies the SHA256 before
    # calling this, so a tampered row is rejected rather than executed. This is
    # the same trust boundary as the prior on-disk .pkl loading this replaces.
    return joblib.load(io.BytesIO(data))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def store(ml_model: MLModel, obj: Any) -> MLModelArtifact:
    """Serialize ``obj`` and persist it as ``ml_model``'s durable artifact.

    Replaces any existing artifact for the row. Returns the saved artifact so
    the caller can copy ``byte_size`` / ``checksum`` onto the ``MLModel`` row if
    it tracks them there too.
    """
    data = _serialize(obj)
    checksum = sha256(data)
    artifact, _ = MLModelArtifact.objects.update_or_create(
        model=ml_model,
        defaults={"data": data, "byte_size": len(data), "checksum": checksum},
    )
    logger.info(
        "Stored model artifact for %s (%d bytes, sha256=%s)",
        ml_model.pk,
        len(data),
        checksum[:12],
    )
    return artifact


def store_bytes(ml_model: MLModel, data: bytes) -> MLModelArtifact:
    """Persist already-serialized joblib bytes as ``ml_model``'s artifact.

    For callers that have just written the model to a file with ``joblib.dump``
    and want the stored bytes to be byte-identical to that file (so a checksum
    computed over the file matches the artifact). ``retrieve`` deserializes the
    same joblib format regardless of whether ``store`` or ``store_bytes`` wrote
    it.
    """
    checksum = sha256(data)
    artifact, _ = MLModelArtifact.objects.update_or_create(
        model=ml_model,
        defaults={"data": data, "byte_size": len(data), "checksum": checksum},
    )
    logger.info(
        "Stored model artifact for %s (%d bytes, sha256=%s)",
        ml_model.pk,
        len(data),
        checksum[:12],
    )
    return artifact


def retrieve(ml_model: MLModel) -> Optional[Any]:
    """Return the deserialized model from durable storage, or ``None``.

    ``None`` means no artifact row exists or the stored bytes failed to load;
    callers treat that as "no model available" and fall back to their legacy
    file path or to rule-based behavior.
    """
    artifact = MLModelArtifact.objects.filter(model=ml_model).first()
    if artifact is None:
        return None
    try:
        data = bytes(artifact.data)
        if artifact.checksum and sha256(data) != artifact.checksum:
            logger.error(
                "Checksum mismatch for model artifact %s; refusing to load",
                ml_model.pk,
            )
            return None
        return _deserialize(data)
    except Exception as exc:  # noqa: BLE001 - any load failure is non-fatal here
        logger.error("Failed to load model artifact %s: %s", ml_model.pk, exc)
        return None


def exists(ml_model: MLModel) -> bool:
    return MLModelArtifact.objects.filter(model=ml_model).exists()


def remove(ml_model: MLModel) -> None:
    """Delete the durable artifact for a row, if any.

    Deleting the ``MLModel`` row cascades to the artifact automatically; this is
    for callers that want to drop the bytes while keeping the metadata row.
    """
    MLModelArtifact.objects.filter(model=ml_model).delete()
