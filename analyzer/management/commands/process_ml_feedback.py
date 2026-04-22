"""
Django management command to process user feedback for ML training.

Usage:
    python manage.py process_ml_feedback
    python manage.py process_ml_feedback --days 7
    python manage.py process_ml_feedback --dry-run
    python manage.py process_ml_feedback --force-all
"""

from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from analyzer.ml.core.feature_extractor import FeatureExtractor
from analyzer.ml.core.feedback_collector import FeedbackCollector
from analyzer.models import Query, QueryFeedback, TrainingData


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
            default=2,
            help="Minimum number of feedback items required per query (default: 2)",
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
            if options["stats_only"]:
                self.show_feedback_statistics(options)
                return

            # Initialize components
            self.feedback_collector = FeedbackCollector()
            self.feature_extractor = FeatureExtractor()

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
                    f"Feedback processing completed. Created training data for {processed_count} queries."
                )
            )

        except Exception as e:
            if options["verbose"]:
                import traceback

                self.stdout.write(self.style.ERROR(traceback.format_exc()))

            raise CommandError(f"Feedback processing failed: {str(e)}")

    def show_feedback_statistics(self, options):
        """Show feedback statistics."""
        self.stdout.write(self.style.SUCCESS("Feedback Statistics"))
        self.stdout.write("=" * 40)

        # Total feedback counts
        total_feedback = QueryFeedback.objects.count()
        positive_feedback = QueryFeedback.objects.filter(is_helpful=True).count()
        negative_feedback = QueryFeedback.objects.filter(is_helpful=False).count()

        self.stdout.write(f"Total feedback items: {total_feedback}")
        self.stdout.write(
            f"Positive feedback: {positive_feedback} ({positive_feedback/total_feedback*100:.1f}%)"
        )
        self.stdout.write(
            f"Negative feedback: {negative_feedback} ({negative_feedback/total_feedback*100:.1f}%)"
        )

        # Recent feedback
        days = options["days"]
        recent_cutoff = timezone.now() - timedelta(days=days)
        recent_feedback = QueryFeedback.objects.filter(
            created_at__gte=recent_cutoff
        ).count()
        self.stdout.write(f"Feedback in last {days} days: {recent_feedback}")

        # Feedback by score agreement
        self.stdout.write("")
        self.stdout.write("Feedback by score agreement:")
        for score in range(1, 6):
            count = QueryFeedback.objects.filter(score_agreement=score).count()
            self.stdout.write(f"  Score {score}: {count} items")

        # Queries with feedback
        queries_with_feedback = (
            Query.objects.filter(queryfeedback__isnull=False).distinct().count()
        )
        total_queries = Query.objects.count()
        self.stdout.write(
            f"Queries with feedback: {queries_with_feedback}/{total_queries} ({queries_with_feedback/total_queries*100:.1f}%)"  # noqa: E501
        )

        # Training data status
        training_data_count = TrainingData.objects.count()
        self.stdout.write(f"Training data samples: {training_data_count}")

        # Queries ready for processing
        queries_ready = self._get_queries_to_process(options)
        self.stdout.write(f"Queries ready for processing: {len(queries_ready)}")

    def _get_queries_to_process(self, options):
        """Get list of queries that need feedback processing."""
        queryset = Query.objects.all()

        # Filter by specific query if provided
        if options["query_id"]:
            queryset = queryset.filter(id=options["query_id"])

        # Filter by date range
        if not options["force_all"]:
            cutoff_date = timezone.now() - timedelta(days=options["days"])
            queryset = queryset.filter(
                queryfeedback__created_at__gte=cutoff_date
            ).distinct()

        # Get queries with sufficient feedback
        queries_to_process = []
        min_feedback = options["min_feedback"]

        for query in queryset:
            feedback_count = query.queryfeedback_set.count()

            if feedback_count >= min_feedback:
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
            feedback_count = query.queryfeedback_set.count()
            total_feedback_items += feedback_count

            self.stdout.write(
                f"Query {query.id} ({query.query_type}): {feedback_count} feedback items"
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
                            f"  Created training data with score {training_data.target_score:.1f} "
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

    def _analyze_feedback_quality(self, options):
        """Analyze feedback quality and user reliability."""
        self.stdout.write(self.style.SUCCESS("Feedback Quality Analysis"))
        self.stdout.write("=" * 40)

        # User reliability analysis
        from django.contrib.auth.models import User
        from django.db.models import Avg, Count

        user_stats = (
            User.objects.filter(queryfeedback__isnull=False)
            .annotate(
                feedback_count=Count("queryfeedback"),
                avg_score_agreement=Avg("queryfeedback__score_agreement"),
            )
            .filter(feedback_count__gte=5)  # Users with at least 5 feedback items
            .order_by("-feedback_count")
        )

        self.stdout.write("Top feedback contributors:")
        for user in user_stats[:10]:
            reliability = self._calculate_user_reliability(user)
            self.stdout.write(
                f"  {user.username}: {user.feedback_count} items, "
                f"avg agreement: {user.avg_score_agreement:.1f}, "
                f"reliability: {reliability:.2f}"
            )

        # Feedback consistency analysis
        self.stdout.write("")
        self.stdout.write("Feedback consistency by query complexity:")

        complexity_ranges = [
            (0, 25, "Simple"),
            (25, 50, "Medium"),
            (50, 75, "Complex"),
            (75, 100, "Very Complex"),
        ]

        for min_complexity, max_complexity, label in complexity_ranges:
            queries = Query.objects.filter(
                estimated_complexity__gte=min_complexity,
                estimated_complexity__lt=max_complexity,
                queryfeedback__isnull=False,
            ).distinct()

            if queries.exists():
                avg_agreement = (
                    QueryFeedback.objects.filter(query__in=queries).aggregate(
                        avg=Avg("score_agreement")
                    )["avg"]
                    or 0
                )

                self.stdout.write(
                    f"  {label} ({min_complexity}-{max_complexity}): {avg_agreement:.1f} avg agreement"
                )

    def _calculate_user_reliability(self, user):
        """Calculate user reliability score based on feedback consistency."""
        # This is a simplified reliability calculation
        # In practice, you might want to use more sophisticated methods

        feedback_items = QueryFeedback.objects.filter(user=user)
        if not feedback_items.exists():
            return 0.0

        # Calculate variance in score agreements (lower variance = more reliable)
        scores = list(feedback_items.values_list("score_agreement", flat=True))
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)

        # Convert to reliability score (0-1, higher is better)
        # Lower variance means higher reliability
        reliability = max(
            0, 1 - variance / 4
        )  # 4 is max possible variance for 1-5 scale

        return reliability
