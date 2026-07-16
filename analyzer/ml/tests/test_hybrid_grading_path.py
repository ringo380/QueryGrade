"""The grading path must actually reach the ML grader when ML is enabled.

analyze_query imported `..ml.hybrid_grader`, but the module lives at
`..ml.core.hybrid_grader`. The resulting ModuleNotFoundError was caught by
a bare `except Exception` and logged as "ML grading failed, falling back to
rule-based", so with ML_ENABLED and ML_HYBRID_GRADING both True the ML path
never ran once -- every grade was rule-based and the only trace was a
warning line. Nothing failed, so nothing surfaced.

These tests assert the path is genuinely taken rather than swallowed.
"""

from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from analyzer.analyzers.base import analyze_query


@override_settings(ML_ENABLED=True, ML_HYBRID_GRADING=True)
class HybridGradingPathTests(TestCase):
    def test_ml_grader_is_actually_invoked(self):
        """The regression: this passed through to rule-based silently."""
        sentinel = ("query-sentinel", "analysis-sentinel")
        grader = MagicMock()
        grader.analyze_query.return_value = sentinel

        with patch(
            "analyzer.ml.core.hybrid_grader.HybridQueryGrader", return_value=grader
        ) as cls:
            result = analyze_query("SELECT 1", "MySQL", use_ml=True)

        cls.assert_called_once()
        grader.analyze_query.assert_called_once_with("SELECT 1", "MySQL", use_ml=True)
        self.assertEqual(result, sentinel)

    def test_import_error_is_not_swallowed(self):
        """A missing module is a packaging bug. Degrading it to a warning is
        what let the path stay broken indefinitely."""
        with patch(
            "analyzer.ml.core.hybrid_grader.HybridQueryGrader",
            side_effect=ImportError("no module named whatever"),
        ):
            with self.assertRaises(ImportError):
                analyze_query("SELECT 1", "MySQL", use_ml=True)

    def test_runtime_failure_still_falls_back(self):
        """A genuine runtime problem (missing artifact, bad model) should
        still degrade to rule-based rather than 500 the grade request."""
        with patch(
            "analyzer.ml.core.hybrid_grader.HybridQueryGrader",
            side_effect=RuntimeError("model artifact missing"),
        ):
            query, analysis = analyze_query("SELECT 1", "MySQL", use_ml=True)

        self.assertIsNotNone(analysis.grade)

    def test_use_ml_false_skips_the_ml_path(self):
        with patch("analyzer.ml.core.hybrid_grader.HybridQueryGrader") as cls:
            analyze_query("SELECT 1", "MySQL", use_ml=False)

        cls.assert_not_called()

    @override_settings(ML_ENABLED=False)
    def test_ml_disabled_skips_the_ml_path(self):
        with patch("analyzer.ml.core.hybrid_grader.HybridQueryGrader") as cls:
            analyze_query("SELECT 1", "MySQL", use_ml=True)

        cls.assert_not_called()
