from django.test import TestCase
from analyzer.parser import parse_mysql_slow_log, parse_mysql_general_log, detect_anomalies, detect_anomalies_general
import pandas as pd
import os

class ParserTestCase(TestCase):
    def setUp(self):
        self.sample_slow_log_path = os.path.join(os.path.dirname(__file__), 'samples', 'mysql-slow-query.log')
        self.sample_general_log_path = os.path.join(os.path.dirname(__file__), 'samples', 'mysql-general-query.log')

    def test_parse_mysql_slow_log(self):
        df = parse_mysql_slow_log(self.sample_slow_log_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) > 0)
        self.assertTrue('timestamp' in df.columns)
        self.assertTrue('query' in df.columns)

    def test_parse_mysql_general_log(self):
        df = parse_mysql_general_log(self.sample_general_log_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) > 0)
        self.assertTrue('timestamp' in df.columns)
        self.assertTrue('query' in df.columns)

    def test_detect_anomalies(self):
        df = parse_mysql_slow_log(self.sample_slow_log_path)
        df = df.head(10)  # Use a smaller subset for testing
        x_scaled = detect_anomalies(df)
        self.assertIsInstance(x_scaled, tuple)
        self.assertEqual(len(x_scaled), 2)

    def test_detect_anomalies_general(self):
        df = parse_mysql_general_log(self.sample_general_log_path)
        df = df.head(10)  # Use a smaller subset for testing
        x_scaled = detect_anomalies_general(df)
        self.assertIsInstance(x_scaled, tuple)
        self.assertEqual(len(x_scaled), 2)
