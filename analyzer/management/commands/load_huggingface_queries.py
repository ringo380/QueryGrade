"""
Pull SQL queries from a HuggingFace dataset, score via the rule-based grader,
and seed TrainingData.

HF datasets are nearly all NL-to-SQL corpora — no public dataset has the
performance grades QueryGrade trains against. We use them as a *query corpus*
for shape coverage: rule-based scoring becomes the synthetic label, and the
sample weight is kept low so real user feedback dominates training.

Default dataset: lamini/spider_text_to_sql (~10k freely-accessible Spider queries).

Examples:
    python manage.py load_huggingface_queries --limit 1000 --dry-run
    python manage.py load_huggingface_queries --limit 5000
    python manage.py load_huggingface_queries --dataset lamini/bird_spider_train_text_to_sql --column output --limit 10000
    python manage.py load_huggingface_queries --clear  # wipe prior HF samples first
"""

import logging
import re

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from analyzer.analyzers.base import QueryGrader
from analyzer.ml.core.feature_extractor import FeatureExtractor
from analyzer.models import TrainingData

logger = logging.getLogger(__name__)

# Default candidates documented in CLAUDE.md (lamini/spider_text_to_sql ~10k,
# lamini/bird_spider_train_text_to_sql 10k–100k, VPCSinfo/odoo-sql-query-dataset).
DEFAULT_DATASET = "lamini/spider_text_to_sql"
DEFAULT_SPLIT = "train"
DEFAULT_COLUMN = "output"  # lamini/spider uses {input, output}; output is the SQL

# Analyzer pipeline is SELECT-optimized (CLAUDE.md: non-SELECT no-ops with a
# performance_notes warning), so filter to queries that lead with SELECT/WITH.
SELECT_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


def _coerce_sql(value, column):
    """Pull a SQL string out of a row that may be dict/str/list."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        # Some datasets nest under {column: {...}}; try common keys.
        for k in (column, "sql", "query", "text", "output"):
            if k in value and isinstance(value[k], str):
                text = value[k]
                break
        else:
            return None
    else:
        return None

    text = text.strip().rstrip(";").strip()
    return text or None


class Command(BaseCommand):
    help = "Seed TrainingData with SQL queries pulled from a HuggingFace dataset"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dataset",
            default=DEFAULT_DATASET,
            help=f"HF dataset id (default: {DEFAULT_DATASET})",
        )
        parser.add_argument(
            "--split",
            default=DEFAULT_SPLIT,
            help=f"Dataset split to load (default: {DEFAULT_SPLIT})",
        )
        parser.add_argument(
            "--column",
            default=DEFAULT_COLUMN,
            help=f"Column containing the SQL string (default: {DEFAULT_COLUMN})",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=1000,
            help="Maximum number of queries to ingest (default: 1000)",
        )
        parser.add_argument(
            "--feedback-weight",
            type=float,
            default=0.3,
            help="Sample weight for HF rows (default: 0.3, vs 1.0 for real feedback)",
        )
        parser.add_argument(
            "--include-non-select",
            action="store_true",
            help="Don't filter out non-SELECT queries (analyzer is SELECT-optimized)",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Delete prior samples for this dataset before seeding",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be created without writing to DB",
        )

    def handle(self, *args, **options):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise CommandError(
                "The `datasets` package is required. Install with:\n"
                "    pip install --break-system-packages datasets\n"
                "(or add to requirements-worker.txt)"
            ) from exc

        dataset_id = options["dataset"]
        split = options["split"]
        column = options["column"]
        limit = options["limit"]
        weight = options["feedback_weight"]
        include_non_select = options["include_non_select"]
        clear = options["clear"]
        dry_run = options["dry_run"]

        source_tag = f"huggingface:{dataset_id}"[:100]

        if clear and not dry_run:
            deleted, _ = TrainingData.objects.filter(
                validation_source=source_tag
            ).delete()
            self.stdout.write(f"Deleted {deleted} prior samples from {source_tag}")

        self.stdout.write(f"Loading {dataset_id} [{split}] from HuggingFace...")
        try:
            ds = load_dataset(dataset_id, split=split, streaming=True)
        except Exception as exc:
            raise CommandError(f"Failed to load dataset: {exc}") from exc

        grader = QueryGrader()
        extractor = FeatureExtractor()
        grade_to_rating = {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "F": 1.0}

        seen_hashes = set()
        scanned = 0
        created = 0
        skipped_non_select = 0
        skipped_dup = 0
        skipped_existing = 0
        errors = 0

        for row in ds:
            if created >= limit:
                break
            scanned += 1

            raw = row.get(column) if isinstance(row, dict) else None
            if raw is None and isinstance(row, dict):
                # Fallback: scan all string-valued columns for one that parses
                for v in row.values():
                    if isinstance(v, str) and SELECT_RE.match(v):
                        raw = v
                        break

            sql = _coerce_sql(raw, column)
            if not sql:
                continue

            if not include_non_select and not SELECT_RE.match(sql):
                skipped_non_select += 1
                continue

            # In-batch dedup before hitting the DB
            sig = hash(sql)
            if sig in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(sig)

            try:
                with transaction.atomic():
                    query_obj, analysis = grader.analyze_query(sql)

                    features = extractor.extract_features(query_obj)
                    if features is None:
                        errors += 1
                        continue

                    if dry_run:
                        if created < 10:
                            self.stdout.write(
                                f"  [dry-run] {analysis.grade} ({analysis.score:.0f}) "
                                f"— {sql[:80]}"
                            )
                        created += 1
                        continue

                    if TrainingData.objects.filter(
                        query=query_obj, validation_source=source_tag
                    ).exists():
                        skipped_existing += 1
                        continue

                    TrainingData.objects.create(
                        query=query_obj,
                        user_grade_avg=grade_to_rating.get(analysis.grade, 3.0),
                        user_grade_count=1,
                        user_grade_stddev=0.0,
                        system_grade=analysis.grade,
                        system_score=analysis.score,
                        features_json=features,
                        target_score=analysis.score,
                        feedback_weight=weight,
                        query_complexity=query_obj.estimated_complexity,
                        table_count=query_obj.table_count,
                        join_count=query_obj.join_count,
                        is_validated=True,
                        validation_source=source_tag,
                    )
                    created += 1

                    if created % 250 == 0:
                        self.stdout.write(
                            f"  ...{created} ingested ({scanned} scanned)"
                        )

            except Exception as exc:
                logger.warning("HF ingest error: %s | sql=%s", exc, sql[:80])
                errors += 1

        verb = "Would create" if dry_run else "Created"
        self.stdout.write(
            self.style.SUCCESS(
                f"\n{verb} {created} samples from {dataset_id} "
                f"(scanned {scanned}, weight={weight})"
            )
        )
        self.stdout.write(
            f"  skipped: non-select={skipped_non_select}, "
            f"in-batch-dup={skipped_dup}, already-loaded={skipped_existing}, "
            f"errors={errors}"
        )

        if not dry_run and created > 0:
            total = TrainingData.objects.count()
            self.stdout.write(f"Total TrainingData records: {total}")
            self.stdout.write(
                "Next: python manage.py train_ml_model --algorithm random_forest --force"
            )
