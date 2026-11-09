"""
Django management command to process user feedback for ML training.

Usage:
    python manage.py process_ml_feedback
    python manage.py process_ml_feedback --days 7
    python manage.py process_ml_feedback --dry-run
    python manage.py process_ml_feedback --force-all
    python manage.py process_ml_feedback --stats-only

Feedback reaches a Query through UserQueryHistory, not directly: a history
row carries either a detailed QueryFeedback (accuracy/usefulness/clarity
ratings) or a simple was_helpful thumbs up/down. FeedbackCollector accepts
both, so the selection and stats here count both -- counting only
QueryFeedback would under-report and disagree with what actually gets
processed.
"""

from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q
from django.utils import timezone

from analyzer.ml.core.feature_extractor import FeatureExtractor
from analyzer.ml.core.feedback_collector import FeedbackCollector
from analyzer.models import Query, QueryFeedback, TrainingData, UserQueryHistory


def _pct(part, whole):
    """Percentage that tolerates an empty dataset.

    Feedback volume is legitimately zero on a fresh or low-traffic install,
    and a stats command must not blow up in exactly the situation you would
    run it to diagnose.
    """
    return f"{(part / whole * 100):.1f}%" if whole else "n/a"


class Command(BaseCommand):
    help = "Process user feedback to create training data for ML models"

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Process feedback from the last N days (default: 30)",
        )

        parser.add_argument(
            "--query-id", type=int, help="Process feedback for a specific query ID"
        )

        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be processed without actually creating training data",
        )

        parser.add_argument(
            "--force-all",
            action="store_true",
            help="Reprocess all feedback, including already processed items",
        )

        parser.add_argument(
            "--min-feedback",
            type=int,
            default=None,
            help=(
                "Minimum feedback items required per query. Defaults to "
                "FeedbackCollector.min_feedback_count, which is the threshold "
                "actually enforced during processing."
            ),
        )

        parser.add_argument(
            "--verbose", action="store_true", help="Enable verbose output"
        )

        parser.add_argument(
            "--stats-only",
            action="store_true",
            help="Show statistics only, do not process feedback",
        )

    def handle(self, *args, **options):
        """Handle the feedback processing command."""
        try:
            # Initialize components
            self.feedback_collector = FeedbackCollector()
            self.feature_extractor = FeatureExtractor()

            # Selecting on a lower threshold than the collector enforces would
            # report queries as "ready" that processing then silently skips.
            if options["min_feedback"] is None:
                options["min_feedback"] = self.feedback_collector.min_feedback_count

            if options["stats_only"]:
                self.show_feedback_statistics(options)
                return

            # Get feedback to process
            queries_to_process = self._get_queries_to_process(options)

            if not queries_to_process:
                self.stdout.write(self.style.WARNING("No queries found for processing"))
                return

            self.stdout.write(f"Found {len(queries_to_process)} queries to process")

            if options["dry_run"]:
                self._show_dry_run_info(queries_to_process, options)
                return

            # Process feedback
            processed_count = self._process_feedback(queries_to_process, options)

            self.stdout.write(
                self.style.SUCCESS(
                    f"Feedback processing completed. Created training data for "
                    f"{processed_count} queries."
                )
            )

        except Exception as e:
            if options["verbose"]:
                import traceback

                self.stdout.write(self.style.ERROR(traceback.format_exc()))

            raise CommandError(f"Feedback processing failed: {str(e)}")

    def _feedback_histories(self):
        """History rows carrying feedback, in either of the two forms."""
        return UserQueryHistory.objects.filter(
            Q(detailed_feedback__isnull=False) | Q(was_helpful__isnull=False)
        )

    def _feedback_count_for(self, query):
        """How many feedback items a query has, counted as the collector counts."""
        return self._feedback_histories().filter(query=query).count()

    def show_feedback_statistics(self, options):
        """Show feedback statistics."""
        self.stdout.write(self.style.SUCCESS("Feedback Statistics"))
        self.stdout.write("=" * 40)

        histories = self._feedback_histories()
        total_feedback = histories.count()
        detailed_count = QueryFeedback.objects.count()
        simple_count = UserQueryHistory.objects.filter(
            detailed_feedback__isnull=True, was_helpful__isnull=False
        ).count()

        self.stdout.write(f"Total feedback items: {total_feedback}")
        self.stdout.write(f"  Detailed (QueryFeedback): {detailed_count}")
        self.stdout.write(f"  Simple (was_helpful):     {simple_count}")

        recommend_yes = QueryFeedback.objects.filter(would_recommend=True).count()
        helpful_yes = UserQueryHistory.objects.filter(was_helpful=True).count()
        self.stdout.write(
            f"Positive: would_recommend={recommend_yes} "
            f"({_pct(recommend_yes, detailed_count)} of detailed), "
            f"was_helpful={helpful_yes} "
            f"({_pct(helpful_yes, simple_count)} of simple)"
        )

        # Recent feedback
        days = options["days"]
        recent_cutoff = timezone.now() - timedelta(days=days)
        recent_feedback = histories.filter(submitted_at__gte=recent_cutoff).count()
        self.stdout.write(f"Feedback in last {days} days: {recent_feedback}")

        # Rating distribution across the three detailed axes.
        self.stdout.write("")
        self.stdout.write("Detailed rating distribution (1-5):")
        for field in ("accuracy_rating", "usefulness_rating", "clarity_rating"):
            counts = [
                QueryFeedback.objects.filter(**{field: score}).count()
                for score in range(1, 6)
            ]
            self.stdout.write(f"  {field:<18} {counts}")

        # Queries with feedback
        queries_with_feedback = histories.values("query").distinct().count()
        total_queries = Query.objects.count()
        self.stdout.write("")
        self.stdout.write(
            f"Queries with feedback: {queries_with_feedback}/{total_queries} "
            f"({_pct(queries_with_feedback, total_queries)})"
        )

        # Training data status, split by origin. is_validated does NOT separate
        # real from synthetic -- the seed rows set it True as well -- so the
        # only honest split is validation_source.
        training_data_count = TrainingData.objects.count()
        synthetic = TrainingData.objects.filter(
            validation_source="synthetic_seed"
        ).count()
        self.stdout.write(
            f"Training data samples: {training_data_count} "
            f"({synthetic} synthetic seed, {training_data_count - synthetic} real)"
        )

        # Queries ready for processing
        queries_ready = self._get_queries_to_process(options)
        self.stdout.write(
            f"Queries ready for processing "
            f"(>= {options['min_feedback']} feedback items): {len(queries_ready)}"
        )

    def _get_queries_to_process(self, options):
        """Get list of queries that need feedback processing."""
        queryset = Query.objects.all()

        # Filter by specific query if provided
        if options["query_id"]:
            queryset = queryset.filter(id=options["query_id"])

        # Filter by date range
        if not options["force_all"]:
            cutoff_date = timezone.now() - timedelta(days=options["days"])
            recent = self._feedback_histories().filter(submitted_at__gte=cutoff_date)
            queryset = queryset.filter(id__in=recent.values("query")).distinct()

        # Get queries with sufficient feedback
        queries_to_process = []
        min_feedback = options["min_feedback"]

        for query in queryset:
            if self._feedback_count_for(query) >= min_feedback:
                # Check if already processed (unless force_all)
                if not options["force_all"]:
                    existing_training_data = TrainingData.objects.filter(
                        query=query
                    ).exists()
                    if existing_training_data:
                        continue

                queries_to_process.append(query)

        return queries_to_process

    def _show_dry_run_info(self, queries_to_process, options):
        """Show information about what would be processed in a dry run."""
        self.stdout.write(
            self.style.WARNING("DRY RUN MODE - No actual processing will occur")
        )
        self.stdout.write("")

        total_feedback_items = 0
        for query in queries_to_process[:10]:  # Show first 10
            feedback_count = self._feedback_count_for(query)
            total_feedback_items += feedback_count

            self.stdout.write(
                f"Query {query.id} ({query.query_type}): "
                f"{feedback_count} feedback items"
            )

        if len(queries_to_process) > 10:
            self.stdout.write(f"... and {len(queries_to_process) - 10} more queries")

        self.stdout.write("")
        self.stdout.write(f"Total queries to process: {len(queries_to_process)}")
        self.stdout.write(f"Estimated feedback items: {total_feedback_items}")

    def _process_feedback(self, queries_to_process, options):
        """Process feedback for the given queries."""
        processed_count = 0
        verbose = options["verbose"]

        for i, query in enumerate(queries_to_process, 1):
            if verbose:
                self.stdout.write(
                    f"Processing query {i}/{len(queries_to_process)}: {query.id}"
                )

            try:
                # Collect feedback and create training data
                training_data = self.feedback_collector.collect_feedback_for_query(
                    query.id
                )

                if training_data:
                    processed_count += 1
                    if verbose:
                        self.stdout.write(
                            f"  Created training data with score "
                            f"{training_data.target_score:.1f} "
                            f"(weight: {training_data.feedback_weight:.2f})"
                        )
                else:
                    if verbose:
                        self.stdout.write(
                            "  No training data created (insufficient feedback)"
                        )

            except Exception as e:
                if verbose:
                    self.stdout.write(
                        self.style.ERROR(
                            f"  Error processing query {query.id}: {str(e)}"
                        )
                    )
                continue

        return processed_count
