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
