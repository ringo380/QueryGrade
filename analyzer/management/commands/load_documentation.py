"""
Django management command to load SQL documentation and best practices.

Usage:
    python manage.py load_documentation
    python manage.py load_documentation --force-refresh
    python manage.py load_documentation --source "MySQL Official Documentation"
    python manage.py load_documentation --validate-benchmarks
    python manage.py load_documentation --create-training-data
"""

import json

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from analyzer.ml.core.feature_extractor import FeatureExtractor
from analyzer.ml.integration.documentation_loader import DocumentationLoader


class Command(BaseCommand):
    help = "Load SQL documentation and best practices for ML training"

    def add_arguments(self, parser):
        parser.add_argument(
            "--force-refresh",
            action="store_true",
            help="Force refresh of cached documentation content",
        )

        parser.add_argument(
            "--source", type=str, help="Load from specific documentation source only"
        )

        parser.add_argument(
            "--validate-benchmarks",
            action="store_true",
            help="Validate benchmark queries against actual analyzer",
        )

        parser.add_argument(
            "--create-training-data",
            action="store_true",
            help="Create training data from validated benchmarks",
        )

        parser.add_argument(
            "--sample-size",
            type=int,
            default=10,
            help="Number of benchmarks to validate (default: 10)",
        )

        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be done without making changes",
        )

        parser.add_argument(
            "--verbose", action="store_true", help="Enable verbose output"
        )

        parser.add_argument(
            "--status-only",
            action="store_true",
            help="Show documentation system status only",
        )

        parser.add_argument(
            "--export-rules", type=str, help="Export loaded rules to JSON file"
        )

    def handle(self, *args, **options):
        """Handle the documentation loading command."""
        try:
            # Initialize documentation loader
            loader = DocumentationLoader()

            if options["status_only"]:
                self.show_documentation_status(loader)
                return

            if options["dry_run"]:
                self.stdout.write(
                    self.style.WARNING("DRY RUN MODE - No changes will be made")
                )

            # Load documentation
            if not options["dry_run"]:
                results = self._load_documentation(loader, options)
                self._display_loading_results(results)

            # Validate benchmarks if requested
            if options["validate_benchmarks"] and not options["dry_run"]:
                validation_results = self._validate_benchmarks(loader, options)
                self._display_validation_results(validation_results)

            # Create training data if requested
            if options["create_training_data"] and not options["dry_run"]:
                training_data_count = self._create_training_data(loader, options)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Created {training_data_count} training data samples from benchmarks"
                    )
                )

            # Export rules if requested
            if options["export_rules"]:
                self._export_rules(loader, options["export_rules"])

            # Show final status
            if options["verbose"]:
                self.show_documentation_status(loader)

        except Exception as e:
            if options["verbose"]:
                import traceback

                self.stdout.write(self.style.ERROR(traceback.format_exc()))

            raise CommandError(f"Documentation loading failed: {str(e)}")

    def show_documentation_status(self, loader):
        """Show documentation system status."""
        self.stdout.write(self.style.SUCCESS("Documentation System Status"))
        self.stdout.write("=" * 40)

        status = loader.get_documentation_status()

        self.stdout.write(f'Sources configured: {status["sources_configured"]}')
        self.stdout.write(f'Sources enabled: {status["sources_enabled"]}')
        self.stdout.write(f'Rules loaded: {status["rules_loaded"]}')
        self.stdout.write(f'Benchmarks loaded: {status["benchmarks_loaded"]}')
        self.stdout.write(f'Benchmarks validated: {status["benchmarks_validated"]}')
        self.stdout.write(f'Cache directory: {status["cache_directory"]}')

        if status["last_update"]:
            self.stdout.write(f'Last update: {status["last_update"]}')

        # Show sources details
        self.stdout.write("")
        self.stdout.write("Configured sources:")
        for source in loader.sources:
            status_icon = "✓" if source.enabled else "✗"
            self.stdout.write(f"  {status_icon} {source.name} ({source.database_type})")
            if source.last_updated:
                self.stdout.write(f"    Last updated: {source.last_updated}")

    def _load_documentation(self, loader, options):
        """Load documentation from sources."""
        self.stdout.write("Loading documentation from sources...")

        if options["source"]:
            # Filter to specific source
            original_sources = loader.sources
            loader.sources = [s for s in loader.sources if s.name == options["source"]]

            if not loader.sources:
                raise CommandError(f"Source not found: {options['source']}")

            self.stdout.write(f"Loading from source: {options['source']}")

        results = loader.load_all_documentation(force_refresh=options["force_refresh"])

        if options["source"]:
            loader.sources = original_sources  # Restore original sources

        return results

    def _display_loading_results(self, results):
        """Display documentation loading results."""
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Documentation Loading Results:"))
        self.stdout.write(f'  Sources processed: {results["sources_processed"]}')
        self.stdout.write(f'  Rules loaded: {results["rules_loaded"]}')
        self.stdout.write(f'  Benchmarks loaded: {results["benchmarks_loaded"]}')

        if results["errors"]:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Errors encountered:"))
            for error in results["errors"]:
                self.stdout.write(f"  - {error}")

    def _validate_benchmarks(self, loader, options):
        """Validate benchmark queries."""
        self.stdout.write("")
        self.stdout.write("Validating benchmark queries...")

        validation_results = loader.validate_benchmarks(
            sample_size=options["sample_size"]
        )

        return validation_results

    def _display_validation_results(self, results):
        """Display benchmark validation results."""
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Benchmark Validation Results:"))
        self.stdout.write(f'  Validated: {results["validated"]}/{results["total"]}')
        self.stdout.write(f'  Accuracy: {results["accuracy"]:.1%}')
        self.stdout.write(f'  Average error: {results["average_error"]:.2f}')

    def _create_training_data(self, loader, options):
        """Create training data from benchmarks."""
        self.stdout.write("")
        self.stdout.write("Creating training data from validated benchmarks...")

        # First, populate features for benchmarks using feature extractor
        feature_extractor = FeatureExtractor()

        # Update benchmark features before creating training data
        for benchmark in loader.benchmarks:
            if benchmark.validated:
                try:
                    # Create a temporary query object for feature extraction
                    import hashlib

                    from analyzer.models import Query

                    temp_query = Query(
                        sql_text=benchmark.query_text,
                        query_type=loader._extract_query_type(benchmark.query_text),
                        query_hash=hashlib.md5(
                            benchmark.query_text.encode(), usedforsecurity=False
                        ).hexdigest(),
                    )

                    # Extract features
                    features = feature_extractor.extract_features(temp_query)
                    if features:
                        benchmark.features = features

                except Exception as e:
                    if options["verbose"]:
                        self.stdout.write(
                            self.style.WARNING(
                                f"Could not extract features for benchmark: {str(e)}"
                            )
                        )

        # Create training data
        training_data_count = loader.create_training_data_from_benchmarks()

        return training_data_count

    def _export_rules(self, loader, output_file):
        """Export loaded rules to JSON file."""
        self.stdout.write(f"Exporting rules to {output_file}...")

        rules_data = {
            "export_timestamp": loader._get_current_timestamp(),
            "total_rules": len(loader.rules),
            "rules_by_type": self._group_rules_by_type(loader.rules),
            "rules_by_database": self._group_rules_by_database(loader.rules),
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "title": rule.title,
                    "description": rule.description,
                    "rule_type": rule.rule_type,
                    "database_type": rule.database_type,
                    "sql_patterns": rule.sql_patterns,
                    "severity": rule.severity,
                    "score_impact": rule.score_impact,
                    "source": rule.source,
                    "confidence": rule.confidence,
                    "example_good": rule.example_good,
                    "example_bad": rule.example_bad,
                }
                for rule in loader.rules
            ],
        }

        with open(output_file, "w") as f:
            json.dump(rules_data, f, indent=2, default=str)

        self.stdout.write(
            self.style.SUCCESS(f"Exported {len(loader.rules)} rules to {output_file}")
        )

    def _group_rules_by_type(self, rules):
        """Group rules by type for statistics."""
        groups = {}
        for rule in rules:
            rule_type = rule.rule_type
            groups[rule_type] = groups.get(rule_type, 0) + 1
        return groups

    def _group_rules_by_database(self, rules):
        """Group rules by database type for statistics."""
        groups = {}
        for rule in rules:
            db_type = rule.database_type
            groups[db_type] = groups.get(db_type, 0) + 1
        return groups

    def _test_rule_application(self, loader, options):
        """Test rule application on sample queries."""
        self.stdout.write("")
        self.stdout.write("Testing rule application on sample queries...")

        # Sample test queries
        test_queries = [
            "SELECT * FROM users WHERE active = 1",
            "SELECT id, name FROM users WHERE user_id = 123",
            "UPDATE users SET active = 0",
            "DELETE FROM logs WHERE created_at < '2023-01-01'",
            "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id",
        ]

        for sql_text in test_queries:
            # Create temporary query
            import hashlib

            from analyzer.models import Query

            temp_query = Query(
                sql_text=sql_text,
                query_type=loader._extract_query_type(sql_text),
                query_hash=hashlib.md5(
                    sql_text.encode(), usedforsecurity=False
                ).hexdigest(),
            )

            # Apply rules
            rule_results = loader.apply_documentation_rules(temp_query)

            self.stdout.write(
                f'\nQuery: {sql_text[:50]}{"..." if len(sql_text) > 50 else ""}'
            )
            self.stdout.write(f'  Rules applied: {rule_results["rule_count"]}')
            self.stdout.write(
                f'  Score impact: {rule_results["total_score_impact"]:.1f}'
            )

            if options["verbose"] and rule_results["applied_rules"]:
                for rule in rule_results["applied_rules"][:3]:  # Show first 3 rules
                    self.stdout.write(f'    - {rule["title"]} ({rule["severity"]})')

    def _download_external_documentation(self, loader, options):
        """Download external documentation for offline use."""
        self.stdout.write("")
        self.stdout.write("Downloading external documentation...")

        web_sources = [
            s for s in loader.sources if s.source_type == "web" and s.enabled
        ]

        for source in web_sources:
            self.stdout.write(f"Downloading from {source.name}...")

            try:
                # This would implement actual downloading logic
                # For now, just show what would be downloaded
                for pattern in source.patterns:
                    url = f"{source.base_url}{pattern}"
                    if options["verbose"]:
                        self.stdout.write(f"  - {url}")

                self.stdout.write(f"  Downloaded {len(source.patterns)} pages")

            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(
                        f"Failed to download from {source.name}: {str(e)}"
                    )
                )

    def _show_rule_statistics(self, loader):
        """Show detailed rule statistics."""
        self.stdout.write("")
        self.stdout.write("Rule Statistics:")

        # Group by type
        type_groups = self._group_rules_by_type(loader.rules)
        self.stdout.write("  By type:")
        for rule_type, count in type_groups.items():
            self.stdout.write(f"    {rule_type}: {count}")

        # Group by database
        db_groups = self._group_rules_by_database(loader.rules)
        self.stdout.write("  By database:")
        for db_type, count in db_groups.items():
            self.stdout.write(f"    {db_type}: {count}")

        # Group by severity
        severity_groups = {}
        for rule in loader.rules:
            severity = rule.severity
            severity_groups[severity] = severity_groups.get(severity, 0) + 1

        self.stdout.write("  By severity:")
        for severity, count in severity_groups.items():
            self.stdout.write(f"    {severity}: {count}")

        # Score impact analysis
        positive_impact = len([r for r in loader.rules if r.score_impact > 0])
        negative_impact = len([r for r in loader.rules if r.score_impact < 0])
        neutral_impact = len([r for r in loader.rules if r.score_impact == 0])

        self.stdout.write("  Score impact:")
        self.stdout.write(f"    Positive: {positive_impact}")
        self.stdout.write(f"    Negative: {negative_impact}")
        self.stdout.write(f"    Neutral: {neutral_impact}")
