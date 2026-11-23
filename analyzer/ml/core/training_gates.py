"""Gates that decide whether the model may (re)train.

The retrain trigger otherwise counts every ``TrainingData`` row, including the
synthetic seed rows that exist only to give the monitoring pipeline an ACTIVE
target. On a zero-traffic install that means the pipeline sees "enough" data and
retrains on pure bootstrap labels, shipping another model with no real
predictive power (issue #92). These helpers count only genuine feedback so
retraining waits for it to accumulate.
"""

import logging

logger = logging.getLogger(__name__)

# validation_source value written to the synthetic bootstrap rows by
# `load_huggingface_queries`. Real feedback rows produced by FeedbackCollector
# carry a different source, so excluding this value isolates real samples.
SYNTHETIC_SEED_SOURCE = "synthetic_seed"


def real_training_sample_count() -> int:
    """Count TrainingData rows that came from real feedback, not the seed."""
    from ...models import TrainingData

    return TrainingData.objects.exclude(validation_source=SYNTHETIC_SEED_SOURCE).count()


def real_feedback_gate():
    """Return ``(ok, real_count, threshold)`` for the real-feedback retrain gate.

    ``ok`` is True when there is enough non-synthetic training data to justify
    retraining. Callers should skip training (or skip auto-deploy) when it is
    False, unless the operator forced the run.
    """
    from django.conf import settings

    threshold = getattr(settings, "ML_MIN_REAL_FEEDBACK_SAMPLES", 25)
    real_count = real_training_sample_count()
    return real_count >= threshold, real_count, threshold
