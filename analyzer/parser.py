from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import re
import pandas as pd
import argparse

def parse_mysql_general_log(file_path):
    """
    Parses a MySQL general log file and extracts relevant information into a DataFrame.

    Args:
        file_path (str): The path to the MySQL general log file.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed log entries with columns for timestamp, thread_id, command_type, 
                      and additional information depending on the command type (e.g., user_host for 'Connect', query for 'Query').
    """
    entries = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # Regular expression to match the general query log entries
            pattern = r'^(\d{4}-\d{2}-\d{2}T[\d:.]+Z)\s+(\d+)\s+(\w+)\s+(.*)'
            match = re.match(pattern, line)

            if match:
                timestamp = match.group(1)
                thread_id = match.group(2)
                command_type = match.group(3)
                rest = match.group(4)

                entry = {
                    'timestamp': timestamp,
                    'thread_id': thread_id,
                    'command_type': command_type,
                }

                if command_type == 'Connect':
                    entry['user_host'] = rest.strip()
                elif command_type == 'Query':
                    entry['query'] = rest.strip()
                else:
                    entry['info'] = rest.strip()

                entries.append(entry)
            else:
                # Handle multiline queries
                if entries and entries[-1]['command_type'] == 'Query':
                    entries[-1]['query'] += ' ' + line

    df = pd.DataFrame(entries)
    return df


def parse_mysql_slow_log(file_path):
    """
    Parses a MySQL slow log file and extracts relevant information into a DataFrame.

    Args:
        file_path (str): The path to the MySQL slow log file.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed log entries with columns for timestamp, user_host, query_time,
                      lock_time, rows_sent, rows_examined, database, set_timestamp, and query.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    entries = []
    entry = {}
    query_lines = []

    for line in lines:
        line = line.strip()

        if line.startswith('# Time:'):
            # Save the previous entry
            if entry and query_lines:
                entry['query'] = ' '.join(query_lines).strip()
                entries.append(entry)
                entry = {}
                query_lines = []

            time_split = line.split('# Time:')
            if len(time_split) > 1:
                entry['timestamp'] = time_split[1].strip()
            else:
                print(f"Error parsing timestamp in line: {line}")

        elif line.startswith('# User@Host:'):
            user_host = line.split('# User@Host:')[1].strip()
            entry['user_host'] = user_host

        elif line.startswith('# Query_time:'):
            # Remove the initial '#' and split
            line_clean = line.lstrip('# ').strip()
            parts: list[str] = line_clean.split()
            try:
                entry['query_time'] = float(parts[1])
                entry['lock_time'] = float(parts[3])
                entry['rows_sent'] = int(parts[5])
                entry['rows_examined'] = int(parts[7])
            except (IndexError, ValueError) as e:
                print(f"Error parsing line: {line}\n{e}")

        elif line.startswith('use '):
            entry['database'] = line.split('use ')[1].strip(';')

        elif line.startswith('SET timestamp='):
            entry['set_timestamp'] = line.split('=')[1].strip(';')

        elif line and not line.startswith('#'):
            query_lines.append(line)

    # Add the last entry
    if entry and query_lines:
        entry['query'] = ' '.join(query_lines).strip()
        entries.append(entry)

    # Convert to DataFrame
    df = pd.DataFrame(entries)
    return df


def detect_anomalies(x_scaled):
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(x_scaled)
    anomaly_scores = model.decision_function(x_scaled)
    anomalies = model.predict(x_scaled)
    return anomalies, anomaly_scores

def detect_anomalies_general(x_scaled):
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(x_scaled)
    anomalies = model.predict(x_scaled)
    anomaly_scores = model.decision_function(x_scaled)
    return anomalies, anomaly_scores

def prepare_features_general(df):
    features = [
        'query_length',
        'num_joins',
        'num_conditions',
        'num_subqueries',
        'is_select',
        'is_update',
        'is_insert',
        'is_delete'
    ]
    x = df[features]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled

def prepare_features(df):
    features = ['query_length', 'num_joins', 'num_conditions', 'num_subqueries', 'query_time', 'rows_ratio']
    x = df[features].copy()  # Use .copy() to avoid SettingWithCopyWarning
    # Handle infinite or NaN values in 'rows_ratio'
    x.loc[:, 'rows_ratio'] = x['rows_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    # Scale the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled

def add_results_to_df(df, anomalies, anomaly_scores):
    df['anomaly'] = anomalies
    df['anomaly_score'] = anomaly_scores
    # Anomalies are labeled as -1
    df_anomalies = df[df['anomaly'] == -1]
    return df, df_anomalies


def display_anomalies(df_anomalies):
    print("Potential Queries for Optimization:")
    print(df_anomalies[['timestamp', 'query', 'query_time', 'anomaly_score']].sort_values('anomaly_score'))


def clean_data(df):
    # Drop entries with missing values in critical fields
    df = df.dropna(subset=['query', 'query_time'])
    return df

def feature_engineering(df):
    df['query_length'] = df['query'].apply(len)
    df['num_joins'] = df['query'].str.upper().str.count('JOIN')
    df['num_conditions'] = df['query'].str.upper().str.count('WHERE') + df['query'].str.upper().str.count('HAVING')
    df['num_subqueries'] = df['query'].str.upper().str.count(r'\(SELECT')
    df['rows_ratio'] = df['rows_examined'] / (df['rows_sent'] + 1)  # Add 1 to avoid division by zero
    return df

def feature_engineering_general_log(df):
    # Ensure all entries in the 'query' column are strings
    df['query'] = df['query'].astype(str)
    
    df['query_length'] = df['query'].apply(len)
    df['num_joins'] = df['query'].str.upper().str.count('JOIN')
    df['num_conditions'] = df['query'].str.upper().str.count('WHERE') + df['query'].str.upper().str.count('HAVING')
    df['num_subqueries'] = df['query'].str.upper().str.count(r'\(SELECT')
    df['is_select'] = df['query'].str.upper().str.startswith('SELECT').astype(int)
    df['is_update'] = df['query'].str.upper().str.startswith('UPDATE').astype(int)
    df['is_insert'] = df['query'].str.upper().str.startswith('INSERT').astype(int)
    df['is_delete'] = df['query'].str.upper().str.startswith('DELETE').astype(int)
    return df

def display_general_anomalies(df_anomalies):
    print("Potential Queries for Optimization:")
    print(df_anomalies[['timestamp', 'query', 'anomaly_score']].sort_values('anomaly_score'))


def process_slow_log(log_file):
    df = parse_mysql_slow_log(log_file)
    df = clean_data(df)
    df = feature_engineering(df)
    x_scaled = prepare_features(df)
    anomalies, anomaly_scores = detect_anomalies(x_scaled)
    df, df_anomalies = add_results_to_df(df, anomalies, anomaly_scores)
    display_anomalies(df_anomalies)

def add_anomaly_results(df, anomalies, anomaly_scores):
    df['anomaly'] = anomalies
    df['anomaly_score'] = anomaly_scores
    df_anomalies = df[df['anomaly'] == -1]
    return df, df_anomalies

def process_general_log(log_file):
    df = parse_mysql_general_log(log_file)
    print(df.head())  # Add this line to inspect the DataFrame
    df = feature_engineering_general_log(df)
    x_scaled = prepare_features_general(df)
    anomalies, anomaly_scores = detect_anomalies_general(x_scaled)
    df, df_anomalies = add_results_to_df(df, anomalies, anomaly_scores)
    display_general_anomalies(df_anomalies)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process MySQL log files.')
    parser.add_argument('log_type', choices=['slow', 'general'], help='Type of the log file (slow or general)')
    parser.add_argument('log_file', help='Path to the log file')

    args = parser.parse_args()

    if args.log_type == 'slow':
        process_slow_log(args.log_file)
    elif args.log_type == 'general':
        process_general_log(args.log_file)