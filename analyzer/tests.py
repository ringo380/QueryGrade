import os

import pandas as pd
from django.test import TestCase

from analyzer.parser import (
    detect_anomalies,
    detect_anomalies_general,
    parse_mysql_general_log,
    parse_mysql_slow_log,
)


class ParserTestCase(TestCase):
    def setUp(self):
        # Sample MySQL logs live at the repo-root `samples/` dir, not under
        # `analyzer/samples/` (no such directory exists).
        repo_root = os.path.dirname(os.path.dirname(__file__))
        self.sample_slow_log_path = os.path.join(
            repo_root, "samples", "mysql-slow-query.log"
        )
        self.sample_general_log_path = os.path.join(
            repo_root, "samples", "mysql-general-query.log"
        )

    def test_parse_mysql_slow_log(self):
        df = parse_mysql_slow_log(self.sample_slow_log_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) > 0)
        self.assertTrue("timestamp" in df.columns)
        self.assertTrue("query" in df.columns)

    def test_parse_mysql_general_log(self):
        df = parse_mysql_general_log(self.sample_general_log_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) > 0)
        self.assertTrue("timestamp" in df.columns)
        self.assertTrue("query" in df.columns)

    def test_detect_anomalies(self):
        df = parse_mysql_slow_log(self.sample_slow_log_path)
        df = df.head(10)  # Use a smaller subset for testing
        # Need to import the required functions for the complete workflow
        from analyzer.parser import clean_data, feature_engineering, prepare_features

        df = clean_data(df)
        df = feature_engineering(df)
        x_scaled = prepare_features(df)
        result = detect_anomalies(x_scaled)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_detect_anomalies_general(self):
        df = parse_mysql_general_log(self.sample_general_log_path)
        df = df.head(10)  # Use a smaller subset for testing
        # Need to import the required functions for the complete workflow
        from analyzer.parser import (
            feature_engineering_general_log,
            prepare_features_general,
        )

        df = feature_engineering_general_log(df)
        x_scaled = prepare_features_general(df)
        result = detect_anomalies_general(x_scaled)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_general_log_parses_non_query_commands_with_no_sql(self):
        """A general log interleaves Connect/Quit lines that carry no SQL.

        Precondition for the featurizing test below: if the sample ever stops
        containing a query-less row, that test would pass without exercising
        the missing-value path at all.
        """
        df = parse_mysql_general_log(self.sample_general_log_path)

        self.assertGreater(
            df["query"].isna().sum(),
            0,
            "sample general log no longer contains a command without SQL",
        )

    def test_feature_engineering_handles_queries_that_are_missing(self):
        """Non-query rows have no SQL, and that must not crash featurizing.

        `astype(str)` alone does NOT cover this: pandas < 3 stringified NaN
        into the literal "nan", but pandas 3 keeps it a missing value, so
        query_length raised "object of type 'float' has no len()" on every
        real general log - including through the upload view.
        """
        from analyzer.parser import feature_engineering_general_log

        df = pd.DataFrame(
            {
                "query": ["SELECT * FROM users WHERE id = 1", None],
                "command_type": ["Query", "Connect"],
            }
        )

        out = feature_engineering_general_log(df)

        self.assertEqual(out["query_length"].tolist(), [32, 0])
        self.assertEqual(out["is_select"].tolist(), [1, 0])
