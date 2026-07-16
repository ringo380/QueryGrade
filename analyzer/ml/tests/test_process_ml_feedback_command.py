"""Tests for the process_ml_feedback management command.

The command had no coverage and had drifted onto a schema that never
existed: it queried QueryFeedback.is_helpful, .score_agreement, .user and
Query.queryfeedback_set, none of which are real fields. Every mode raised
CommandError on the first ORM call, so the documented path from real user
feedback to TrainingData could not run at all.

Feedback reaches a Query only through UserQueryHistory, in two forms
(detailed QueryFeedback ratings, or a simple was_helpful flag), and
FeedbackCollector accepts both. These tests drive the real command against
real rows so a schema drift fails here instead of at the operator's prompt.
"""

from io import StringIO

from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase

from analyzer.ml.core.feedback_collector import FeedbackCollector
from analyzer.models import (
    Query,
    QueryAnalysis,
    QueryFeedback,
    TrainingData,
    UserQueryHistory,
)


class ProcessMlFeedbackCommandTests(TestCase):
    def setUp(self):
        self.query = Query.objects.create(
            sql_text="SELECT * FROM users WHERE id = 1",
            query_type="SELECT",
            query_hash="cmd_test_hash",
            estimated_complexity=25,
            table_count=1,
            join_count=0,
            where_conditions=1,
            subquery_count=0,
        )
        QueryAnalysis.objects.create(
            query=self.query,
            grade="C",
            score=75.0,
            issues_found=[{"type": "SELECT_STAR", "severity": "medium"}],
            recommendations=[{"type": "SELECT_SPECIFIC", "priority": "medium"}],
            performance_notes="Query uses SELECT *",
        )

    def add_detailed_feedback(self, n, query=None):
        """n users leave full ratings on a query."""
        for i in range(n):
            user = User.objects.create_user(
                username=f"detailed{i}{id(query or self.query)}", password="pw"
            )
            history = UserQueryHistory.objects.create(
                user=user, query=query or self.query, database_type="MySQL"
            )
            QueryFeedback.objects.create(
                user_history=history,
                accuracy_rating=4,
                usefulness_rating=4,
                clarity_rating=4,
                would_recommend=True,
            )

    def add_simple_feedback(self, n):
        """n users leave only a was_helpful thumbs up."""
        for i in range(n):
            user = User.objects.create_user(username=f"simple{i}", password="pw")
            UserQueryHistory.objects.create(
                user=user,
                query=self.query,
                database_type="MySQL",
                was_helpful=True,
            )

    def run_cmd(self, *args):
        out = StringIO()
        call_command("process_ml_feedback", *args, stdout=out, stderr=out)
        return out.getvalue()

    # --- the regression: every mode used to raise CommandError ---

    def test_stats_only_runs_with_no_feedback_at_all(self):
        """Zero feedback is the normal state of a fresh install, and is
        exactly when someone runs --stats-only to find out why."""
        output = self.run_cmd("--stats-only")

        self.assertIn("Total feedback items: 0", output)
        # A percentage over an empty set must not raise ZeroDivisionError.
        self.assertIn("n/a", output)

    def test_dry_run_and_plain_run_survive_no_feedback(self):
        self.assertIn("No queries found", self.run_cmd("--dry-run"))
        self.assertIn("No queries found", self.run_cmd())

    # --- counting the real schema ---

    def test_stats_counts_detailed_feedback(self):
        self.add_detailed_feedback(3)

        output = self.run_cmd("--stats-only")

        self.assertIn("Total feedback items: 3", output)
        self.assertIn("Detailed (QueryFeedback): 3", output)
        self.assertIn("Queries with feedback: 1/1", output)

    def test_stats_counts_simple_was_helpful_feedback(self):
        """was_helpful is feedback too: FeedbackCollector accepts it, so
        counting only QueryFeedback would under-report."""
        self.add_simple_feedback(2)

        output = self.run_cmd("--stats-only")

        self.assertIn("Total feedback items: 2", output)
        self.assertIn("Simple (was_helpful):     2", output)

    def test_stats_separates_synthetic_seed_from_real_training_data(self):
        """is_validated is True on the synthetic seed rows too, so only
        validation_source can tell real training data from bootstrap."""
        TrainingData.objects.create(
            query=self.query,
            user_grade_avg=4.0,
            user_grade_count=3,
            system_grade="C",
            system_score=75.0,
            is_validated=True,
            validation_source="synthetic_seed",
        )

        output = self.run_cmd("--stats-only")

        self.assertIn("1 synthetic seed, 0 real", output)

    # --- selection has to agree with what processing enforces ---

    def test_selection_threshold_defaults_to_the_collector_threshold(self):
        """Selecting on a lower bar than the collector enforces reports
        queries as ready that processing then silently drops."""
        collector_min = FeedbackCollector().min_feedback_count
        self.add_detailed_feedback(collector_min - 1)

        output = self.run_cmd("--stats-only")

        self.assertIn(
            f"(>= {collector_min} feedback items): 0", output
        )

    def test_query_with_enough_feedback_is_found_and_processed(self):
        self.add_detailed_feedback(FeedbackCollector().min_feedback_count)

        # Precondition: without this, "0 processed" below would pass
        # vacuously on a query nothing could ever select.
        self.assertIn("Found 1 queries to process", self.run_cmd("--dry-run"))
        self.assertFalse(TrainingData.objects.filter(query=self.query).exists())

        output = self.run_cmd("--verbose")

        self.assertIn("Created training data for 1 queries", output)
        self.assertTrue(TrainingData.objects.filter(query=self.query).exists())

    def test_already_processed_query_is_skipped(self):
        self.add_detailed_feedback(FeedbackCollector().min_feedback_count)
        self.run_cmd()
        self.assertTrue(TrainingData.objects.filter(query=self.query).exists())

        # Second run must not re-process the same query.
        self.assertIn("No queries found", self.run_cmd())
